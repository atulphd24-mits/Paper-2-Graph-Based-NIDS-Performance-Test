import pandas as pd

FEATURE_GROUPS = {
    "TCP_Flags": ["flag", "syn", "ack", "rst", "fin", "urg", "psh"],
    "Flow": ["flow", "duration", "rate", "iat", "number"],
    "Packet": ["packet", "header", "tot", "bytes"],
    "Protocol": [
        "protocol", "tcp", "udp", "icmp",
        "http", "https", "dns", "ssh",
        "arp", "ipv", "llc"
    ],
    "Statistical": [
        "min", "max", "avg", "std",
        "variance", "covariance",
        "weight", "radius", "magnit"
    ]
}


class FeatureEngineer:

    def __init__(self, df, label_column="label"):

        self.df = df.copy()
        self.label_column = label_column

        self.X = None
        self.y = None

        self.feature_groups = {}
        self.log = []

    def split_features_labels(self):

        if self.label_column not in self.df.columns:
            raise ValueError(f"{self.label_column} not found.")

        self.X = self.df.drop(columns=[self.label_column])
        self.y = self.df[self.label_column]

    def group_features(self):

        groups = {key: [] for key in FEATURE_GROUPS}
        groups["Other"] = []

        assigned = set()

        for col in self.X.columns:

            name = col.lower()

            matched = False

            for group, keywords in FEATURE_GROUPS.items():

                if any(keyword in name for keyword in keywords):

                    groups[group].append(col)
                    assigned.add(col)
                    matched = True
                    break

            if not matched:
                groups["Other"].append(col)

        self.feature_groups = groups
        self.log.append("✓ Feature grouping completed")

    def summary(self):

        print("=" * 60)
        print("FEATURE ENGINEERING REPORT")
        print("=" * 60)

        print(f"Samples  : {self.X.shape[0]}")
        print(f"Features : {self.X.shape[1]}")
        print()

        for group, cols in self.feature_groups.items():
            print(group)
            print("-" * 40)
            print(cols)
            print()

    def engineer(self):

        self.split_features_labels()
        self.group_features()

        return self.X, self.y