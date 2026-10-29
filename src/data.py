from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
TARGET = "attack_cat"
LEAKY_COLS = ["id", "label"]
CATEGORICAL_COLS = ["proto", "service", "state"]
FEATURE_GROUPS = {
    "volume": ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
               "sloss", "dloss", "smean", "dmean"],
    "timing": ["sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"],
    "ttl": ["sttl", "dttl", "ct_state_ttl"],
    "tcp_header": ["swin", "dwin", "stcpb", "dtcpb"],
    "app_content": ["trans_depth", "response_body_len", "is_ftp_login", "ct_ftp_cmd",
                    "ct_flw_http_mthd"],
    "conn_history": ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                     "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                     "is_sm_ips_ports"],
}
NUMERIC_COLS = [col for cols in FEATURE_GROUPS.values() for col in cols]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
CLASS_ORDER = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
               "Analysis", "Backdoor", "Shellcode", "Worms"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_ORDER)}
N_CLASSES = len(CLASS_ORDER)


def load_data():
    train = pd.read_csv(RAW_DIR / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(RAW_DIR / "UNSW_NB15_testing-set.csv")
    for df in (train, test):
        missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df["service"] = df["service"].replace("-", "none")
    return train, test


def deduplicate(df):
    return df.drop_duplicates(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)


def split_X_y(df):
    return df[FEATURE_COLS].reset_index(drop=True), df[TARGET].map(CLASS_TO_ID).to_numpy(dtype=int)


def load_model_data():
    train, test = load_data()
    X_train, y_train = split_X_y(deduplicate(train))
    X_test, y_test = split_X_y(test)
    return X_train, y_train, X_test, y_test
