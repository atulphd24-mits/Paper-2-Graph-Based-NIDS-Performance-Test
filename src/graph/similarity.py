"""
Hybrid Similarity Calculator

Author : Atul Chauhan
Project : SOC Threat Detection Framework

Similarity =
Feature +
Protocol +
Temporal +
Port +
Flow Behaviour

Design notes
------------
* Temporal similarity decays exponentially with a time constant ``tau``
  (milliseconds). Normalising by the whole capture span made almost every
  pair look simultaneous (median similarity 0.994).
* Port similarity compares the destination (service) port only by default.
  Source ports are ephemeral, so matching them is mostly noise.
* Flow similarity compares log-scaled duration / bytes / packets with a
  distance kernel. Cosine on raw values was dominated by bytes and returned
  ~1.0 for nearly all pairs.
"""

import numpy as np


def _cosine_rows(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)

    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), 0.0)


class SimilarityCalculator:

    def __init__(
        self,
        feature_weight=0.45,
        protocol_weight=0.20,
        temporal_weight=0.15,
        port_weight=0.10,
        flow_weight=0.10,
        temporal_tau=60_000.0,
        src_port_weight=0.0
    ):

        total = (
            feature_weight +
            protocol_weight +
            temporal_weight +
            port_weight +
            flow_weight
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError("Similarity weights must sum to 1.")

        if not 0.0 <= src_port_weight <= 1.0:
            raise ValueError("src_port_weight must be between 0 and 1.")

        self.feature_weight = feature_weight
        self.protocol_weight = protocol_weight
        self.temporal_weight = temporal_weight
        self.port_weight = port_weight
        self.flow_weight = flow_weight

        self.temporal_tau = temporal_tau
        self.src_port_weight = src_port_weight

    # --------------------------------------------------------
    # Feature Similarity
    # --------------------------------------------------------

    def feature_similarity(self, x1, x2):

        return float(_cosine_rows(x1, x2))

    # --------------------------------------------------------
    # Protocol Similarity
    # --------------------------------------------------------

    def protocol_similarity(self, p1, p2):

        return 1.0 if p1 == p2 else 0.0

    # --------------------------------------------------------
    # Port Similarity
    # --------------------------------------------------------

    def port_similarity(
        self,
        src1,
        dst1,
        src2,
        dst2
    ):

        dst_score = 1.0 if dst1 == dst2 else 0.0
        src_score = 1.0 if src1 == src2 else 0.0

        return (
            (1.0 - self.src_port_weight) * dst_score +
            self.src_port_weight * src_score
        )

    # --------------------------------------------------------
    # Temporal Similarity
    # --------------------------------------------------------

    def temporal_similarity(
        self,
        t1,
        t2,
        max_difference=None
    ):

        diff = abs(float(t1) - float(t2))

        if self.temporal_tau is None:
            # Legacy behaviour: linear decay over the whole capture span.
            if not max_difference:
                raise ValueError("max_difference is required when temporal_tau is None")
            return max(0.0, 1 - diff / max_difference)

        return float(np.exp(-diff / self.temporal_tau))

    # --------------------------------------------------------
    # Flow Behaviour Similarity
    # --------------------------------------------------------

    def flow_similarity(
        self,
        duration1,
        bytes1,
        packets1,
        duration2,
        bytes2,
        packets2
    ):

        v1 = np.log1p(np.maximum([duration1, bytes1, packets1], 0))
        v2 = np.log1p(np.maximum([duration2, bytes2, packets2], 0))

        distance = np.linalg.norm(v1 - v2) / np.sqrt(len(v1))

        return float(np.exp(-distance))

    # --------------------------------------------------------
    # Hybrid Similarity
    # --------------------------------------------------------

    def hybrid_similarity(
        self,

        feature1,
        feature2,

        protocol1,
        protocol2,

        time1,
        time2,

        src_port1,
        dst_port1,
        src_port2,
        dst_port2,

        duration1,
        bytes1,
        packets1,

        duration2,
        bytes2,
        packets2,
        max_time=None,
        max_difference=None,
        **kwargs
    ):

        if max_time is None:
            max_time = max_difference

        feature = self.feature_similarity(
            feature1,
            feature2
        )

        protocol = self.protocol_similarity(
            protocol1,
            protocol2
        )

        temporal = self.temporal_similarity(
            time1,
            time2,
            max_time
        )

        port = self.port_similarity(
            src_port1,
            dst_port1,
            src_port2,
            dst_port2
        )

        flow = self.flow_similarity(
            duration1,
            bytes1,
            packets1,
            duration2,
            bytes2,
            packets2
        )

        return self._combine(feature, protocol, temporal, port, flow)

    # --------------------------------------------------------
    # Vectorised Hybrid Similarity (many pairs at once)
    # --------------------------------------------------------

    def hybrid_similarity_batch(
        self,
        features_a,
        features_b,
        protocol_a,
        protocol_b,
        time_a,
        time_b,
        src_port_a,
        dst_port_a,
        src_port_b,
        dst_port_b,
        flow_a,
        flow_b
    ):
        """
        Compute all similarity components for aligned arrays of pairs.

        ``flow_a`` / ``flow_b`` are (n_pairs, 3) arrays of
        duration, bytes and packets.
        """

        feature = _cosine_rows(features_a, features_b)

        protocol = (np.asarray(protocol_a) == np.asarray(protocol_b)).astype(float)

        diff = np.abs(np.asarray(time_a, dtype=float) - np.asarray(time_b, dtype=float))
        temporal = np.exp(-diff / self.temporal_tau)

        port = (
            (1.0 - self.src_port_weight)
            * (np.asarray(dst_port_a) == np.asarray(dst_port_b))
            + self.src_port_weight
            * (np.asarray(src_port_a) == np.asarray(src_port_b))
        ).astype(float)

        log_a = np.log1p(np.maximum(np.asarray(flow_a, dtype=float), 0))
        log_b = np.log1p(np.maximum(np.asarray(flow_b, dtype=float), 0))
        flow = np.exp(-np.linalg.norm(log_a - log_b, axis=1) / np.sqrt(log_a.shape[1]))

        return self._combine(feature, protocol, temporal, port, flow)

    def _combine(self, feature, protocol, temporal, port, flow):

        hybrid = (

            self.feature_weight * feature +

            self.protocol_weight * protocol +

            self.temporal_weight * temporal +

            self.port_weight * port +

            self.flow_weight * flow

        )

        return {

            "feature_similarity": feature,

            "protocol_similarity": protocol,

            "temporal_similarity": temporal,

            "port_similarity": port,

            "flow_similarity": flow,

            "hybrid_similarity": hybrid

        }
