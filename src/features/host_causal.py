"""
Past-only (causal) host-behaviour features.

Replaces the tumbling-window aggregates of ``src.graph.host_features`` for the
paper-2 final runs. The earlier version assigned every flow to a calendar bin
(``time // window``) and joined the statistics of the whole bin back to it, so a
flow saw up to one window length of *future* traffic.

Here, for a flow starting at time t, every statistic uses only flows of the
same key (source host, destination host, or source-destination pair) that
start in the window (t - w, t] and come no later in a fixed stable order:

* the flow itself is counted;
* flows starting at exactly the same millisecond are counted only if they come
  earlier in the stable order (the row order of the input table; for BigFlow the
  stream order: file number, then row within the file);
* flows starting after t are never used.

Changing or deleting flows that come later never changes the features of the
flow at t. The windows are built over the full stream of the source (all flows
of every split): no labels are used, and a SIEM sees all past traffic.

The statistics, names and order match ``HostBehaviorFeatures``, so downstream
code is unchanged:

    {view}{window}_flows, _peers, _dst_ports, _syn_only_ratio, _no_reply_ratio,
    _tiny_flow_ratio, _mean_log_bytes, _port_entropy, _port_ratio
    (pair views have no _peers column)

Implementation: rows are sorted by (key, time) and scanned once per view and
window with two pointers; per-peer and per-port counters are updated
incrementally, so the cost is O(n) per view and window. Features are written
only for the requested target rows, so memory stays small even for 27.5M flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

SYN, ACK = 2, 16
STAT_NAMES = ["flows", "peers", "dst_ports", "syn_only_ratio", "no_reply_ratio",
              "tiny_flow_ratio", "mean_log_bytes", "port_entropy", "port_ratio"]


@njit(cache=True)
def _scan(keys, times, peers, ports, syn, noreply, tiny, logb, target, window,
          n_peers, n_ports, out):
    """
    keys/times/...: arrays sorted by (key, time).
    target[i] >= 0: row of ``out`` that receives the features of sorted row i.
    out columns: flows, peers, ports, syn, noreply, tiny, logbytes, entropy, port_ratio.
    """
    n = keys.shape[0]
    peer_cnt = np.zeros(n_peers, np.int64)
    port_cnt = np.zeros(n_ports, np.int64)
    g0 = 0
    while g0 < n:
        g1 = g0
        while g1 < n and keys[g1] == keys[g0]:
            g1 += 1
        lo = g0
        hi = g0
        n_distinct_peers = 0
        n_distinct_ports = 0
        sum_clogc = 0.0
        s_syn = 0.0
        s_nr = 0.0
        s_tiny = 0.0
        s_logb = 0.0
        for i in range(g0, g1):
            t = times[i]
            # Rows are sorted by (key, time, stable order): add rows up to and including i,
            # i.e. earlier flows and same-time flows that come earlier in the stable order.
            while hi <= i:
                c = port_cnt[ports[hi]]
                if c > 0:
                    sum_clogc -= c * np.log2(c)
                else:
                    n_distinct_ports += 1
                port_cnt[ports[hi]] = c + 1
                sum_clogc += (c + 1) * np.log2(c + 1)
                if peer_cnt[peers[hi]] == 0:
                    n_distinct_peers += 1
                peer_cnt[peers[hi]] += 1
                s_syn += syn[hi]
                s_nr += noreply[hi]
                s_tiny += tiny[hi]
                s_logb += logb[hi]
                hi += 1
            # drop flows that start at or before t - window
            while lo < hi and times[lo] <= t - window:
                c = port_cnt[ports[lo]]
                sum_clogc -= c * np.log2(c)
                port_cnt[ports[lo]] = c - 1
                if c - 1 > 0:
                    sum_clogc += (c - 1) * np.log2(c - 1)
                else:
                    n_distinct_ports -= 1
                peer_cnt[peers[lo]] -= 1
                if peer_cnt[peers[lo]] == 0:
                    n_distinct_peers -= 1
                s_syn -= syn[lo]
                s_nr -= noreply[lo]
                s_tiny -= tiny[lo]
                s_logb -= logb[lo]
                lo += 1
            row = target[i]
            if row >= 0:
                m = hi - lo
                out[row, 0] = m
                out[row, 1] = n_distinct_peers
                out[row, 2] = n_distinct_ports
                out[row, 3] = s_syn / m
                out[row, 4] = s_nr / m
                out[row, 5] = s_tiny / m
                out[row, 6] = s_logb / m
                ent = np.log2(m) - sum_clogc / m
                out[row, 7] = ent if ent > 0.0 else 0.0
                out[row, 8] = n_distinct_ports / m
        # reset counters for the next group
        for j in range(lo, hi):
            peer_cnt[peers[j]] -= 1
            port_cnt[ports[j]] -= 1
        g0 = g1


class CausalHostFeatures:
    """Past-only host features for selected target rows of a flow table."""

    def __init__(
        self,
        windows_ms=(1_000, 10_000, 60_000),
        src_column="ipv4_src_addr",
        dst_column="ipv4_dst_addr",
        dst_port_column="l4_dst_port",
        time_column="flow_start_milliseconds",
        flags_column="client_tcp_flags",
        bytes_column="in_bytes",
        packets_column="in_pkts",
        reply_packets_column="out_pkts",
    ):
        self.windows_ms = tuple(windows_ms)
        self.src_column = src_column
        self.dst_column = dst_column
        self.dst_port_column = dst_port_column
        self.time_column = time_column
        self.flags_column = flags_column
        self.bytes_column = bytes_column
        self.packets_column = packets_column
        self.reply_packets_column = reply_packets_column

    @staticmethod
    def _name(window_ms):
        return f"{window_ms // 1000}s" if window_ms >= 1000 else f"{window_ms}ms"

    def columns(self):
        cols = []
        for w in self.windows_ms:
            name = self._name(w)
            for view in ("src", "dst", "pair"):
                stats = [s for s in STAT_NAMES if not (view == "pair" and s == "peers")]
                cols += [f"{view}{name}_{s}" for s in stats]
        return cols

    def transform(self, df, target_mask=None):
        """
        Features for the rows where ``target_mask`` is True (all rows if None),
        computed from all rows of ``df``. Returns a DataFrame indexed like the
        target rows of ``df``.
        """
        n = len(df)
        target_mask = np.ones(n, bool) if target_mask is None else np.asarray(target_mask, bool)
        target_rows = np.flatnonzero(target_mask)
        target_of_row = np.full(n, -1, np.int64)
        target_of_row[target_rows] = np.arange(len(target_rows))

        times = pd.to_numeric(df[self.time_column]).to_numpy(np.int64)
        stable = np.arange(n)  # stable order = row order of the input table
        src = pd.factorize(df[self.src_column])[0].astype(np.int64)
        dst = pd.factorize(df[self.dst_column])[0].astype(np.int64)
        port = pd.factorize(df[self.dst_port_column])[0].astype(np.int64)
        flags = df[self.flags_column].fillna(0).to_numpy(np.int64)
        syn = (((flags & SYN) > 0) & ((flags & ACK) == 0)).astype(np.float64)
        noreply = (df[self.reply_packets_column].fillna(0).to_numpy() == 0).astype(np.float64)
        tiny = (df[self.packets_column].to_numpy() <= 2).astype(np.float64)
        logb = np.log1p(np.clip(df[self.bytes_column].to_numpy(np.float64), 0, None))

        n_hosts = int(max(src.max(), dst.max()) + 1) if n else 1
        n_ports = int(port.max() + 1) if n else 1
        pair = src * n_hosts + dst

        views = {
            "src": (src, dst, n_hosts),
            "dst": (dst, src, n_hosts),
            "pair": (pair, port, n_ports),  # peers == ports for a pair
        }
        blocks = []
        for w in self.windows_ms:
            name = self._name(w)
            for view, (key, peer, n_peer) in views.items():
                order = np.lexsort((stable, times, key))
                out = np.zeros((len(target_rows), len(STAT_NAMES)))
                _scan(key[order], times[order], peer[order], port[order], syn[order],
                      noreply[order], tiny[order], logb[order], target_of_row[order],
                      np.int64(w), n_peer, n_ports, out)
                cols = [f"{view}{name}_{s}" for s in STAT_NAMES]
                block = pd.DataFrame(out, columns=cols)
                if view == "pair":
                    block = block.drop(columns=[f"pair{name}_peers"])
                blocks.append(block)
        result = pd.concat(blocks, axis=1)
        result.index = df.index[target_rows]
        return result
