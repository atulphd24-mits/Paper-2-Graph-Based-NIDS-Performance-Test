"""
Host Behaviour Features

Author : Atul Chauhan
Project : SOC Threat Detection Framework

Scanning and (D)DoS are properties of a host's behaviour across many
flows, not of a single flow. These features summarise, for each flow,
what its source and destination hosts were doing in the same time
window of the IP communication graph:

For each view (source host, destination host, source->destination pair)
and each window size: flow count, distinct peers, distinct destination
ports, destination-port entropy and ratio, SYN-without-ACK ratio,
no-reply ratio, share of tiny flows, mean log bytes.

No labels are used. Features are computed on the complete traffic table
(not just the sampled rows), because a deployed sensor would see all of it.

The module also provides ``build_ip_node_features`` - a corrected version
of the DeepDRAC IP-node aggregation, whose averages previously divided a
sum of two means by the flow count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SYN = 2
ACK = 16


def _entropy_from_counts(levels, counts):
    """Shannon entropy of a count distribution per group of index levels."""
    totals = counts.groupby(level=levels).transform("sum")
    p = counts / totals
    return (-(p * np.log2(p))).groupby(level=levels).sum()


class HostBehaviorFeatures:
    """
    Multi-scale host behaviour features.

    For every window size, three views are computed per flow:

    * ``src``  - the source host's out-going behaviour
    * ``dst``  - the destination host's in-coming behaviour
    * ``pair`` - the source -> destination conversation (port sweeps show
      up here: one pair, many destination ports, no replies)

    Short windows matter: in BigFlow, scanners also send benign traffic,
    so long windows dilute the scan signal with normal activity.
    """

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
        window_ms=None,
    ):
        # ``window_ms`` is kept for backwards compatibility (single window).
        self.windows_ms = (window_ms,) if window_ms else tuple(windows_ms)
        self.src_column = src_column
        self.dst_column = dst_column
        self.dst_port_column = dst_port_column
        self.time_column = time_column
        self.flags_column = flags_column
        self.bytes_column = bytes_column
        self.packets_column = packets_column
        self.reply_packets_column = reply_packets_column

    def _window_name(self, window_ms):
        return f"{window_ms // 1000}s" if window_ms >= 1000 else f"{window_ms}ms"

    def _group_features(self, df, keys, prefix, peer_column):
        grouped = df.groupby(keys, sort=False)

        features = pd.DataFrame(
            {
                f"{prefix}_flows": grouped.size(),
                f"{prefix}_peers": grouped[peer_column].nunique(),
                f"{prefix}_dst_ports": grouped[self.dst_port_column].nunique(),
                f"{prefix}_syn_only_ratio": grouped["_syn_only"].mean(),
                f"{prefix}_no_reply_ratio": grouped["_no_reply"].mean(),
                f"{prefix}_tiny_flow_ratio": grouped["_tiny"].mean(),
                f"{prefix}_mean_log_bytes": grouped["_log_bytes"].mean(),
            }
        )

        port_counts = df.groupby(keys + [self.dst_port_column], sort=False).size()
        features[f"{prefix}_port_entropy"] = _entropy_from_counts(
            list(range(len(keys))), port_counts
        )
        features[f"{prefix}_port_ratio"] = (
            features[f"{prefix}_dst_ports"] / features[f"{prefix}_flows"]
        )

        return features

    def transform(self, df):
        """Return one row of host features per flow in ``df`` (same index)."""
        flags = df[self.flags_column].fillna(0).astype(np.int64)
        data = df[
            [self.src_column, self.dst_column, self.dst_port_column, self.time_column]
        ].assign(
            _syn_only=((flags & SYN) > 0) & ((flags & ACK) == 0),
            _no_reply=df[self.reply_packets_column].fillna(0) == 0,
            _tiny=df[self.packets_column] <= 2,
            _log_bytes=np.log1p(df[self.bytes_column].clip(lower=0)),
        )

        parts = []
        for window_ms in self.windows_ms:
            name = self._window_name(window_ms)
            data["_window"] = (data[self.time_column] // window_ms).astype(np.int64)

            views = {
                f"src{name}": ([self.src_column, "_window"], self.dst_column),
                f"dst{name}": ([self.dst_column, "_window"], self.src_column),
                f"pair{name}": (
                    [self.src_column, self.dst_column, "_window"],
                    self.dst_port_column,
                ),
            }
            for prefix, (keys, peer) in views.items():
                features = self._group_features(data, keys, prefix, peer)
                if prefix.startswith("pair"):
                    # peers == dst_ports for a pair; drop the duplicate.
                    features = features.drop(columns=[f"{prefix}_peers"])
                parts.append(data[keys].join(features, on=keys)[features.columns])

        return pd.concat(parts, axis=1).fillna(0.0)


def build_ip_node_features(flows, node_to_id, num_nodes=None):
    """
    Correct per-IP node features for the DeepDRAC communication graph.

    Averages are flow-weighted over all incoming and outgoing flows of a
    host (sum / count), instead of (mean_out + mean_in) / total_count.
    Heavy-tailed totals are log1p-compressed.
    """
    num_nodes = num_nodes or len(node_to_id)

    flows = flows.reset_index(drop=True)
    src = flows["ipv4_src_addr"].map(node_to_id)
    dst = flows["ipv4_dst_addr"].map(node_to_id)

    if src.isna().any() or dst.isna().any():
        raise ValueError("Some IP addresses cannot be mapped to graph nodes.")

    src = src.astype(int).to_numpy()
    dst = dst.astype(int).to_numpy()

    def total(nodes, values):
        return np.bincount(nodes, weights=values, minlength=num_nodes)

    ones = np.ones(len(flows))
    out_count = total(src, ones)
    in_count = total(dst, ones)
    flow_count = out_count + in_count

    # In NetFlow, in_* is client->server (sent by the source host) and
    # out_* is server->client (sent by the destination host).
    sent_bytes = total(src, flows["in_bytes"].to_numpy(float)) + total(
        dst, flows["out_bytes"].to_numpy(float)
    )
    received_bytes = total(src, flows["out_bytes"].to_numpy(float)) + total(
        dst, flows["in_bytes"].to_numpy(float)
    )
    sent_pkts = total(src, flows["in_pkts"].to_numpy(float)) + total(
        dst, flows["out_pkts"].to_numpy(float)
    )
    received_pkts = total(src, flows["out_pkts"].to_numpy(float)) + total(
        dst, flows["in_pkts"].to_numpy(float)
    )

    def flow_mean(column):
        values = flows[column].to_numpy(float)
        summed = total(src, values) + total(dst, values)
        return np.where(flow_count > 0, summed / np.maximum(flow_count, 1), 0.0)

    endpoints = pd.DataFrame(
        {"node": np.concatenate([src, dst]), "peer": np.concatenate([dst, src])}
    )
    distinct_peers = (
        endpoints.groupby("node")["peer"].nunique()
        .reindex(range(num_nodes), fill_value=0)
        .to_numpy(float)
    )

    features = pd.DataFrame(
        {
            "out_flow_count": out_count,
            "in_flow_count": in_count,
            "total_flow_count": flow_count,
            "bytes_sent": sent_bytes,
            "bytes_received": received_bytes,
            "pkts_sent": sent_pkts,
            "pkts_received": received_pkts,
            "avg_flow_duration": flow_mean("flow_duration_milliseconds"),
            "avg_packet_size": flow_mean("longest_flow_pkt"),
            "avg_throughput": flow_mean("src_to_dst_avg_throughput"),
            "distinct_peers": distinct_peers,
        }
    )

    return np.log1p(features.clip(lower=0))
