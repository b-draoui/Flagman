"""
src/data.py -- one place that loads UNSW-NB15 and names the columns.

This file is "plumbing", so it is given to you complete. Every notebook imports
from here, which guarantees that all of you load and clean the data in exactly
the same way (a requirement of the benchmarking protocol).

Read it once, top to bottom, so you know what it does.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
# This file lives in Flagman/src/, so the project root is one folder up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- columns
TARGET = "attack_cat"

# Columns that must NEVER be given to a model:
#   id    -> a row counter. It carries no network information, and if the file
#            is ordered by class it lets a model "cheat" (check this in notebook 02).
#   label -> 0 = Normal, 1 = attack. It is just a summary of the target itself.
LEAKY_COLS = ["id", "label"]

CATEGORICAL_COLS = ["proto", "service", "state"]

# The 39 numeric features, grouped by what they measure.
# The groups are reused for (1) the feature table in the paper's Dataset section
# and (2) the feature-group ablation experiments of the contribution track.
FEATURE_GROUPS = {
    # how much traffic, and how fast
    "volume": ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
               "sloss", "dloss", "smean", "dmean"],
    # timing between packets and of the TCP handshake
    "timing": ["sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"],
    # time-to-live values
    "ttl": ["sttl", "dttl", "ct_state_ttl"],
    # TCP header fields
    "tcp_header": ["swin", "dwin", "stcpb", "dtcpb"],
    # application-layer content (HTTP / FTP)
    "app_content": ["trans_depth", "response_body_len", "is_ftp_login", "ct_ftp_cmd",
                    "ct_flw_http_mthd"],
    # counts over the last 100 connections
    "conn_history": ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                     "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                     "is_sm_ips_ports"],
}
NUMERIC_COLS = [col for group in FEATURE_GROUPS.values() for col in group]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS  # 39 + 3 = 42 model inputs

# A fixed class order so every table and plot lists the classes the same way
# (most frequent -> rarest in the training set).
CLASS_ORDER = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
               "Analysis", "Backdoor", "Shellcode", "Worms"]


# ---------------------------------------------------------------- functions
def load_data():
    """Return (train, test) as DataFrames, lightly cleaned."""
    train = pd.read_csv(RAW_DIR / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(RAW_DIR / "UNSW_NB15_testing-set.csv")

    for df in (train, test):
        _check_columns(df)
        # "-" means "no application-layer service was recognised". It is NOT a
        # missing value: it covers ~54% of training rows and is strongly tied to
        # the class (all Shellcode flows have it). So we keep it as a category of
        # its own and only give it a readable name.
        df["service"] = df["service"].replace("-", "none")

    return train, test


def split_X_y(df):
    """Inputs X (the 42 features) and target y (attack_cat)."""
    return df[FEATURE_COLS].copy(), df[TARGET].copy()


def _check_columns(df):
    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"These expected columns are missing from the CSV: {missing}")
