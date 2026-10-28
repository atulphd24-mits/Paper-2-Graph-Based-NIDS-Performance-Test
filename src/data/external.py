"""
Loaders for the external validation datasets.

Both datasets are mapped onto the column conventions used for BigFlow, so the
same splitting, feature and evaluation code applies:

    flow_start_milliseconds, ipv4_src_addr, ipv4_dst_addr, l4_src_port,
    l4_dst_port (when available), label (0/1), attack (family name, "Benign"),
    source (dataset name)

UNSW-NB15 (original release)
    Four raw CSV files, 2,540,044 records, 47 Argus/Bro features plus
    source/destination IP and port and start/end time (seconds). Supports
    every analysis of the study.

CIC-IoT2023 (feature-CSV release)
    169 CSV files of 46 packet-window features and a fine-grained label.
    No timestamps, addresses or ports, so time-based splits, host features and
    graph models cannot be applied.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dataset_dir(name):
    path = Path(Config().get_dataset_path(name))
    return path if path.is_absolute() else PROJECT_ROOT / path


# ----------------------------------------------------------------------
# UNSW-NB15 (original Argus / Bro features)
# ----------------------------------------------------------------------

UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

UNSW_IDENTIFIERS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]
UNSW_CATEGORICAL = ["proto", "state", "service"]

UNSW_ATTACK_NAMES = {"Backdoors": "Backdoor"}


def unsw_dir():
    return _dataset_dir("unsw_nb15_raw")


def unsw_raw_files():
    return sorted(glob.glob(str(unsw_dir() / "UNSW-NB15_[0-9].csv")))


def _port(series):
    """Ports are mostly decimal, a few hexadecimal ('0x000b') or '-'."""
    text = series.astype(str).str.strip()
    hexa = text.str.startswith("0x")
    out = pd.to_numeric(text.where(~hexa), errors="coerce")
    out[hexa] = text[hexa].apply(lambda v: int(v, 16))
    return out.fillna(-1).astype(np.int64)


def load_unsw():
    """All 2.54M records of the original UNSW-NB15 release, BigFlow conventions."""
    parts = [
        pd.read_csv(path, header=None, names=UNSW_COLUMNS, encoding="utf-8-sig",
                    low_memory=False)
        for path in unsw_raw_files()
    ]
    df = pd.concat(parts, ignore_index=True)

    attack = df["attack_cat"].fillna("").astype(str).str.strip().replace(UNSW_ATTACK_NAMES)
    df["attack"] = np.where(attack == "", "Benign", attack)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    for col in UNSW_CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().astype("category").cat.codes

    df["flow_start_milliseconds"] = pd.to_numeric(df["stime"], errors="coerce") * 1000
    df["flow_end_milliseconds"] = pd.to_numeric(df["ltime"], errors="coerce") * 1000
    df["ipv4_src_addr"] = df["srcip"].astype(str).str.strip()
    df["ipv4_dst_addr"] = df["dstip"].astype(str).str.strip()
    df["l4_src_port"] = _port(df["sport"])
    df["l4_dst_port"] = _port(df["dsport"])
    df["source"] = "UNSW-NB15 (original)"

    numeric = [c for c in UNSW_COLUMNS if c not in UNSW_IDENTIFIERS + ["attack_cat", "label"]]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.drop(columns=UNSW_IDENTIFIERS + ["attack_cat"])


def unsw_behaviour_columns():
    return [c for c in UNSW_COLUMNS if c not in UNSW_IDENTIFIERS + ["attack_cat", "label"]]


def load_unsw_official_split():
    """The widely used pre-split training (175,341) and testing (82,332) sets."""
    folder = unsw_dir() / "Traning_and_Testing"
    frames = {}
    for name, file in (("train", "UNSW_NB15_training-set.csv"), ("test", "UNSW_NB15_testing-set.csv")):
        df = pd.read_csv(folder / file, encoding="utf-8-sig")
        df["attack"] = df["attack_cat"].astype(str).str.strip().replace(
            {"Normal": "Benign", **UNSW_ATTACK_NAMES})
        for col in UNSW_CATEGORICAL:
            df[col] = df[col].astype(str)
        frames[name] = df.drop(columns=["id", "attack_cat"])

    # Encode categoricals with one vocabulary for both sets.
    for col in UNSW_CATEGORICAL:
        vocabulary = pd.Categorical(pd.concat([frames["train"][col], frames["test"][col]])).categories
        for df in frames.values():
            df[col] = pd.Categorical(df[col], categories=vocabulary).codes
    features = [c for c in frames["train"].columns if c not in ("attack", "label")]
    return frames["train"], frames["test"], features


# ----------------------------------------------------------------------
# CIC-IoT2023 (feature CSVs)
# ----------------------------------------------------------------------

CICIOT_BENIGN = "BenignTraffic"


def ciciot_dir():
    return _dataset_dir("cic_iot")


def ciciot_files():
    return sorted(glob.glob(str(ciciot_dir() / "*.csv")))


def ciciot_family_map():
    mapping = pd.read_csv(PROJECT_ROOT / "notebooks/cic_iot_label_family_mapping.csv")
    return dict(zip(mapping["label"], mapping["attack_family"]))


def read_ciciot_file(path):
    df = pd.read_csv(path)
    df["attack"] = df["label"].replace({CICIOT_BENIGN: "Benign"})
    df["label"] = (df["attack"] != "Benign").astype(int)
    df["source"] = "CIC-IoT2023"
    return df


def ciciot_behaviour_columns(columns):
    return [c for c in columns if c not in ("label", "attack", "source", "file_id", "file_row")]
