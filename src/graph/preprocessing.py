"""
Flow Feature Transformer

Author : Atul Chauhan
Project : SOC Threat Detection Framework

Turns raw NetFlow columns into a representation suitable for
distance-based graph construction and neural models:

1. Drops identifiers and absolute timestamps (they leak the capture
   schedule instead of describing behaviour).
2. log1p-compresses heavy-tailed counters (bytes, packets, throughput).
3. One-hot encodes categorical fields (destination port, protocol, L7).
4. Expands TCP flag bitmasks into individual flag bits.
5. Standard-scales everything using statistics from the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Columns that identify a flow or its position in the capture rather than
# describing behaviour. Using them as features lets a model memorise when or
# between whom an attack happened.
LEAKY_COLUMNS = [
    "flow_start_milliseconds",
    "flow_end_milliseconds",
    "ipv4_src_addr",
    "ipv4_dst_addr",
    "l4_src_port",
    "dns_query_id",
    "label",
    "attack",
]

DEFAULT_CATEGORICAL = ["l4_dst_port", "protocol", "l7_proto"]

DEFAULT_FLAG_COLUMNS = ["tcp_flags", "client_tcp_flags", "server_tcp_flags"]

TCP_FLAG_BITS = {
    "fin": 1,
    "syn": 2,
    "rst": 4,
    "psh": 8,
    "ack": 16,
    "urg": 32,
}


class FlowFeatureTransformer:
    """Fit on training rows, then transform any rows consistently."""

    def __init__(
        self,
        numeric_columns=None,
        categorical_columns=None,
        flag_columns=None,
        max_categories=20,
        skew_threshold=2.0,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = (
            DEFAULT_CATEGORICAL
            if categorical_columns is None
            else list(categorical_columns)
        )
        self.flag_columns = (
            DEFAULT_FLAG_COLUMNS
            if flag_columns is None
            else list(flag_columns)
        )
        self.max_categories = max_categories
        self.skew_threshold = skew_threshold

        self.log_columns_ = []
        self.categories_ = {}
        self.scaler_ = StandardScaler()
        self.feature_names_ = []

    # ---------------------------------------------------
    # Column resolution
    # ---------------------------------------------------

    def _resolve_numeric(self, df):
        if self.numeric_columns is not None:
            columns = list(self.numeric_columns)
        else:
            columns = list(df.select_dtypes(include=np.number).columns)

        # Flag bitmasks are expanded into bits, not treated as magnitudes.
        excluded = (
            set(LEAKY_COLUMNS)
            | set(self.categorical_columns)
            | set(self.flag_columns)
        )
        return [col for col in columns if col not in excluded]

    # ---------------------------------------------------
    # Encoding helpers
    # ---------------------------------------------------

    def _encode(self, df):
        parts = []

        numeric = df[self.numeric_].astype(float).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in self.log_columns_:
            numeric[col] = np.log1p(numeric[col].clip(lower=0))
        parts.append(numeric)

        for col in self.flag_columns_:
            values = df[col].fillna(0).astype(np.int64)
            bits = pd.DataFrame(
                {
                    f"{col}_{name}": ((values & bit) > 0).astype(float)
                    for name, bit in TCP_FLAG_BITS.items()
                },
                index=df.index,
            )
            parts.append(bits)

        for col, categories in self.categories_.items():
            values = df[col]
            onehot = pd.DataFrame(
                {
                    f"{col}={cat}": (values == cat).astype(float)
                    for cat in categories
                },
                index=df.index,
            )
            onehot[f"{col}=other"] = (~values.isin(categories)).astype(float)
            parts.append(onehot)

        return pd.concat(parts, axis=1)

    # ---------------------------------------------------
    # Public API
    # ---------------------------------------------------

    def fit(self, df):
        self.numeric_ = self._resolve_numeric(df)
        self.flag_columns_ = [
            col for col in self.flag_columns
            if col in df.columns
            and (self.numeric_columns is None or col in self.numeric_columns)
        ]

        numeric = df[self.numeric_].astype(float)
        self.log_columns_ = [
            col
            for col in self.numeric_
            if numeric[col].min() >= 0
            and abs(numeric[col].skew()) > self.skew_threshold
        ]

        self.categories_ = {
            col: list(df[col].value_counts().index[: self.max_categories])
            for col in self.categorical_columns
            if col in df.columns
        }

        encoded = self._encode(df)
        self.feature_names_ = list(encoded.columns)
        self.scaler_.fit(encoded.to_numpy())

        return self

    def transform(self, df):
        encoded = self._encode(df)[self.feature_names_]
        return self.scaler_.transform(encoded.to_numpy())

    def fit_transform(self, df, fit_rows=None):
        """Fit on ``fit_rows`` (e.g. the training mask) and transform all rows."""
        fit_df = df if fit_rows is None else df.loc[fit_rows]
        return self.fit(fit_df).transform(df)
