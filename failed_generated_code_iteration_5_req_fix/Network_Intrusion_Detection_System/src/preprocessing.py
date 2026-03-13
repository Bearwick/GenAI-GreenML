import pandas as pd
from sklearn.preprocessing import StandardScaler

# Raw NSL-KDD columns
RAW_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

def preprocess_data(df, fit=True, scaler=None, columns=None):
    df = df.copy()

    # Assign columns if missing
    if df.shape[1] <= len(RAW_COLUMNS):
        df.columns = RAW_COLUMNS[:df.shape[1]]

    # Binary label
    if "label" in df.columns:
        df["label"] = df["label"].apply(lambda x: 0 if x=="normal" else 1)

    # Separate numeric and categorical
    num_cols = [c for c in df.columns if c not in CATEGORICAL_COLS + ["label","difficulty"]]
    X_num = df[num_cols].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS]) if CATEGORICAL_COLS else pd.DataFrame()

    # Combine
    X = pd.concat([X_num, X_cat], axis=1)

    # Align with training columns
    if columns is not None:
        # Keep only columns seen in training
        X = X[[col for col in X.columns if col in columns]]
        # Add missing columns with 0
        for col in columns:
            if col not in X.columns:
                X[col] = 0
        # Reorder exactly
        X = X[columns]

    # Scale
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, df["label"] if "label" in df.columns else None, scaler, X.columns.tolist()
    else:
        X_scaled = scaler.transform(X)
        return X_scaled, df["label"] if "label" in df.columns else None
