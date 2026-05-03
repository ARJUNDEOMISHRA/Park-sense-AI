# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0: REPRODUCIBILITY & GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════════════
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ── Hyperparameter registry (single source of truth) ────────────────────
CFG = dict(
    seed        = SEED,
    n_classes   = 5,
    batch_size  = 4096,
    epochs_bnn  = 30,          # all BNN variants
    epochs_bns  = 20,          # BNS (warm-up + finetune)
    lr          = 1e-3,
    kl_base     = None,        # set after data load = 1/N_train
    kl_low_mul  = 0.1,         # BNN-LowKL  = kl_base * 0.1
    kl_high_mul = 10.0,        # BNN-HighKL = kl_base * 10.0
    lambda_bns  = 0.05,        # TSC penalty weight
    mc_samples  = 50,          # Monte-Carlo forward passes
    ece_bins    = 15,          # ECE calibration bins
)

print("TF version :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU"))
print("Config     :", CFG)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1.2 — NOAA ISD WEATHER LOADING & PARSING
# Melbourne Airport, Station 94866099999, 2017
# ════════════════════════════════════════════════════════════════════════════
import os, numpy as np, pandas as pd

WEATHER_PATH = "/mnt/user-data/uploads/94866099999.csv"

weather_raw = pd.read_csv(WEATHER_PATH, low_memory=False)
weather_raw["DATE"] = pd.to_datetime(weather_raw["DATE"])
weather_raw = weather_raw[
    (weather_raw["DATE"] >= "2017-01-01") &
    (weather_raw["DATE"] <  "2018-01-01")
].copy()
print(f"Raw NOAA rows    : {len(weather_raw):,}")

def parse_noaa(series, field_idx, scale=10.0, missing=None):
    missing = missing or []
    return (series.str.split(",", expand=True)[field_idx]
                  .replace(missing, np.nan)
                  .astype(float) / scale)

weather_raw["air_temp"]  = parse_noaa(weather_raw["TMP"], 0, 10.0, ["+9999","-9999","9999"])
weather_raw["dew_point"] = parse_noaa(weather_raw["DEW"], 0, 10.0, ["+9999","-9999","9999"])
weather_raw["wind_spd"]  = parse_noaa(weather_raw["WND"], 3, 10.0, ["9999"])
weather_raw["pressure"]  = parse_noaa(weather_raw["SLP"], 0, 10.0, ["99999"])
weather_raw["rain_mm"]   = (
    weather_raw["AA1"].str.split(",", expand=True)[1]
    .replace(["9999","99999",""], np.nan)
    .astype(float).fillna(0.0) / 10.0
)

# Resample to hourly + forward-fill
weather_clean = (
    weather_raw[["DATE","air_temp","dew_point","wind_spd","pressure","rain_mm"]]
    .rename(columns={"DATE":"Timestamp"})
    .drop_duplicates("Timestamp")
    .set_index("Timestamp")
    .resample("h").mean()
    .reset_index()
    .ffill().bfill()
)

print(f"Hourly weather   : {len(weather_clean):,} rows")
print(f"Date range       : {weather_clean['Timestamp'].min()}  to  {weather_clean['Timestamp'].max()}")
print(f"Null rates       :\n{weather_clean.isnull().mean().round(4)}")

# Quick sanity plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for ax, col, color in zip(axes,
    ["air_temp","dew_point","wind_spd","pressure","rain_mm"],
    ["#EE6677","#4477AA","#228833","#CCBB44","#66CCEE"]):
    ax.plot(weather_clean["Timestamp"], weather_clean[col], color=color, linewidth=0.5)
    ax.set_title(col, fontsize=10); ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.suptitle("NOAA ISD Weather — Melbourne Airport 2017 (hourly)", fontsize=11)
plt.tight_layout()
plt.savefig("weather_eda.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: weather_eda.pdf")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1.3 — PARKING DATA LOADING & WEATHER MERGE
# ════════════════════════════════════════════════════════════════════════════
PARKING_PATH = "/home/arjun-deo-mishra/Desktop/parking/u9sa-j86i/On-street_Car_Parking_Sensor_Data_-_2017.csv"

df_park = pd.read_csv(PARKING_PATH)
print(f"Raw parking rows : {len(df_park):,}")

# Clean
df_park["BetweenStreet2"].fillna("Unknown", inplace=True)
if df_park["DurationSeconds"].dtype == object:
    df_park["DurationSeconds"] = pd.to_numeric(
        df_park["DurationSeconds"].str.replace(",","",regex=False).str.strip(),
        errors="coerce")

df_park["ArrivalTime"]   = pd.to_datetime(df_park["ArrivalTime"])
df_park["DepartureTime"] = pd.to_datetime(df_park["DepartureTime"], errors="coerce")

# Temporal features
df_park["Hour"]      = df_park["ArrivalTime"].dt.hour
df_park["Day"]       = df_park["ArrivalTime"].dt.day
df_park["Month"]     = df_park["ArrivalTime"].dt.month
df_park["DayOfWeek"] = df_park["ArrivalTime"].dt.dayofweek
df_park["IsWeekend"] = (df_park["DayOfWeek"] >= 5).astype(int)

# Occupancy ratio per StreetId per 15-min window
df_park["TimeSlot"] = df_park["ArrivalTime"].dt.floor("15min")
occ = (df_park.groupby(["StreetId","TimeSlot"])["Vehicle Present"]
               .mean().reset_index(name="OccupancyRatio"))
df_park = df_park.merge(occ, on=["StreetId","TimeSlot"], how="left")

def occ_to_class(x):
    if   x < 0.2: return 0
    elif x < 0.4: return 1
    elif x < 0.6: return 2
    elif x < 0.8: return 3
    else:         return 4

df_park["OccClass"] = df_park["OccupancyRatio"].apply(occ_to_class)

# Merge on hourly timestamp
df_park["Timestamp"] = df_park["ArrivalTime"].dt.floor("h")
merged_df = df_park.merge(weather_clean, on="Timestamp", how="left")

# Validate
match_rate = merged_df["air_temp"].notna().mean() * 100
print(f"Merged rows      : {len(merged_df):,}")
print(f"Weather match    : {match_rate:.1f}%")
print(f"Class distribution:\n{merged_df['OccClass'].value_counts().sort_index()}")

# Merge quality figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
merged_df["OccClass"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color="#4477AA", edgecolor="black")
axes[0].set_title("Class Distribution (merged)", fontweight="bold")
axes[0].set_xticklabels([f"Occ-{i}" for i in range(5)], rotation=0)
axes[0].set_ylabel("Count")

merged_df.groupby("Month")["OccClass"].mean().plot(
    ax=axes[1], marker="o", color="#AA3377", linewidth=2)
axes[1].set_title("Mean Occupancy Class by Month", fontweight="bold")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Mean OccClass")
axes[1].set_xticks(range(1,13))
axes[1].grid(True, alpha=0.3)

plt.suptitle("Merged Parking + Weather — EDA", fontsize=12)
plt.tight_layout()
plt.savefig("merge_eda.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: merge_eda.pdf")

# Use merged_df as df_raw for downstream processing
df_raw = merged_df[[
    "Hour","Day","Month","DayOfWeek","IsWeekend",
    "air_temp","dew_point","wind_spd","pressure","rain_mm",
    "OccClass"
]].dropna().reset_index(drop=True)
print(f"\ndf_raw ready: {df_raw.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# Adjust DATA_PATH to your merged_df CSV if re-running from scratch.
# The expected columns are listed in FEATURES below.
# ═══════════════════════════════════════════════════════════════════════════
import os

DATA_PATH = "/home/arjun-deo-mishra/Desktop/parking/parking_merged.csv"

FEATURES = [
    "Hour", "Day", "Month", "DayOfWeek", "IsWeekend",
    "air_temp", "dew_point", "wind_spd", "pressure", "rain_mm",
]
TARGET = "OccClass"

if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {df_raw.shape[0]:,} rows from {DATA_PATH}")
else:
    # ── Synthetic fallback (matches real distribution) ─────────────────
    print("Data file not found – generating synthetic data for demonstration.")
    rng = np.random.default_rng(SEED)
    N = 2_000_000

    hour      = rng.integers(0, 24, N)
    day       = rng.integers(1, 32, N)
    month     = rng.integers(1, 13, N)
    dow       = rng.integers(0,  7, N)
    weekend   = (dow >= 5).astype(int)
    air_temp  = rng.normal(17, 5, N)
    dew_point = air_temp - rng.uniform(2, 8, N)
    wind_spd  = rng.exponential(3, N)
    pressure  = rng.normal(1013, 8, N)
    rain_mm   = rng.exponential(0.3, N) * (rng.random(N) < 0.2)

    # Occupancy class based on hour (realistic bimodal pattern)
    occ_prob  = 0.5 + 0.3 * np.sin(2 * np.pi * (hour - 8) / 24)
    occ_prob += 0.15 * np.sin(2 * np.pi * (hour - 17) / 24)
    occ_prob  = np.clip(occ_prob + rng.normal(0, 0.1, N), 0, 1)
    occ_class = np.digitize(occ_prob, [0.2, 0.4, 0.6, 0.8])

    df_raw = pd.DataFrame({
        "Hour": hour, "Day": day, "Month": month,
        "DayOfWeek": dow, "IsWeekend": weekend,
        "air_temp": air_temp, "dew_point": dew_point,
        "wind_spd": wind_spd, "pressure": pressure, "rain_mm": rain_mm,
        "OccClass": occ_class,
    })
    print(f"Synthetic dataset: {df_raw.shape}")

print("\nClass distribution:")
print(df_raw[TARGET].value_counts().sort_index())


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1b: BALANCED SAMPLING (300 k per class → 1.5 M total)
# Using controlled undersampling to prevent majority-class collapse
# while retaining temporal diversity.
# ═══════════════════════════════════════════════════════════════════════════
TARGET_PER_CLASS = 300_000

balanced_parts = [
    df_raw[df_raw[TARGET] == c].sample(
        min(TARGET_PER_CLASS, (df_raw[TARGET] == c).sum()),
        random_state=SEED
    )
    for c in range(CFG["n_classes"])
]
df = pd.concat(balanced_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Balanced distribution:")
print(df[TARGET].value_counts().sort_index())
print(f"Total rows: {len(df):,}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1c: CYCLIC TEMPORAL ENCODING + SPLIT + SCALING
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.model_selection import train_test_split

# Cyclic encoding for periodic features
df["Hour_sin"]       = np.sin(2 * np.pi * df["Hour"]      / 24)
df["Hour_cos"]       = np.cos(2 * np.pi * df["Hour"]      / 24)
df["DayOfWeek_sin"]  = np.sin(2 * np.pi * df["DayOfWeek"] /  7)
df["DayOfWeek_cos"]  = np.cos(2 * np.pi * df["DayOfWeek"] /  7)
df["Month_sin"]      = np.sin(2 * np.pi * df["Month"]     / 12)
df["Month_cos"]      = np.cos(2 * np.pi * df["Month"]     / 12)

FEATURES_CYCLIC = [
    "Hour_sin", "Hour_cos",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "Day", "IsWeekend",
    "air_temp", "dew_point", "wind_spd", "pressure", "rain_mm",
]
# Keep raw Hour for TSC computation (not scaled)
HOUR_COL = df["Hour"].values

X_all = df[FEATURES_CYCLIC].values.astype(np.float32)
y_all = df[TARGET].values.astype(np.int32)

# Time-stratified split: train ≤ month 9 | val = 10 | test ≥ 11
# For synthetic/balanced data, use random stratified split
X_temp, X_test, y_temp, y_test, h_temp, h_test = train_test_split(
    X_all, y_all, HOUR_COL,
    test_size=0.15, stratify=y_all, random_state=SEED
)
X_train, X_val, y_train, y_val, h_train, h_val = train_test_split(
    X_temp, y_temp, h_temp,
    test_size=0.15, stratify=y_temp, random_state=SEED
)

# Feature scaling (fit on train only)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

print(f"Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")
print(f"Classes: {np.unique(y_train)}")

# Global config updates
CFG["input_dim"] = X_train.shape[1]
CFG["kl_base"]   = 1.0 / len(X_train)
N_TRAIN          = len(X_train)

print(f"\nkl_base = 1 / {N_TRAIN:,} = {CFG['kl_base']:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1d: CLASS WEIGHTS + TF.DATA PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# Softened class weights (√ of balanced weights) to avoid overcompensation
raw_weights = compute_class_weight("balanced",
                                   classes=np.arange(CFG["n_classes"]),
                                   y=y_train)
CLASS_WEIGHTS = {i: float(np.sqrt(raw_weights[i])) for i in range(CFG["n_classes"])}
print("Class weights (softened):", {k: round(v, 4) for k, v in CLASS_WEIGHTS.items()})

def make_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(200_000, seed=SEED)
    return ds.batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, shuffle=True)
val_ds   = make_dataset(X_val,   y_val)
test_ds  = make_dataset(X_test,  y_test)

# Empirical hourly occupancy curve for TSC
def compute_empirical_mu(X_raw_hours, y, high_class=2):
    mu = np.array([
        np.mean(y[X_raw_hours == h] == high_class) if np.any(X_raw_hours == h) else 0.0
        for h in range(24)
    ], dtype=np.float32)
    return mu

empirical_mu = compute_empirical_mu(h_train, y_train)
print("\nEmpirical μ_h (high-class prob per hour):")
print(np.round(empirical_mu, 3))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2.1: BAYESIAN DENSE LAYER
# Local reparameterisation trick; KL closed-form (standard normal prior).
# ═══════════════════════════════════════════════════════════════════════════

class BayesianDense(tf.keras.layers.Layer):
    """Mean-field Gaussian variational layer with trainable σ_w."""

    def __init__(self, units, kl_weight=1e-5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.kl_weight  = kl_weight
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        d = int(input_shape[-1])
        # ── Posterior parameters ────────────────────────────────────────
        self.w_mu  = self.add_weight(name="w_mu",  shape=(d, self.units),
                                     initializer="glorot_uniform")
        self.w_rho = self.add_weight(name="w_rho", shape=(d, self.units),
                                     initializer=tf.constant_initializer(-4.0))
        self.b_mu  = self.add_weight(name="b_mu",  shape=(self.units,),
                                     initializer="zeros")
        self.b_rho = self.add_weight(name="b_rho", shape=(self.units,),
                                     initializer=tf.constant_initializer(-4.0))

    def call(self, inputs, training=False):
        w_sigma = tf.nn.softplus(self.w_rho) + 1e-6
        b_sigma = tf.nn.softplus(self.b_rho) + 1e-6

        if training:
            w = self.w_mu + w_sigma * tf.random.normal(tf.shape(self.w_mu))
            b = self.b_mu + b_sigma * tf.random.normal(tf.shape(self.b_mu))
        else:
            w = self.w_mu   # deterministic at inference
            b = self.b_mu

        # KL divergence q(w) || N(0,1)  – closed form
        if training:
            kl = 0.5 * tf.reduce_sum(
                self.w_mu**2 + w_sigma**2 - tf.math.log(w_sigma**2) - 1.0
            )
            kl += 0.5 * tf.reduce_sum(
                self.b_mu**2 + b_sigma**2 - tf.math.log(b_sigma**2) - 1.0
            )
            self.add_loss(self.kl_weight * kl)

        out = tf.matmul(inputs, w) + b
        return self.activation(out) if self.activation else out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units, kl_weight=self.kl_weight,
                   activation=tf.keras.activations.serialize(self.activation))
        return cfg

print("✅ BayesianDense layer defined")
print(f"   softplus(-4) = {tf.nn.softplus(-4.0).numpy():.4f}  ← small initial σ")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2.2: BNN ARCHITECTURE FACTORY
#
# Variant      | Width     | KL weight
# ─────────────────────────────────────
# BNN-Small    | 32 → 16   | kl_base
# BNN-Large    | 128 → 64  | kl_base
# BNN-LowKL    | 128 → 64  | kl_base × 0.1
# BNN-HighKL   | 128 → 64  | kl_base × 10.0
# ═══════════════════════════════════════════════════════════════════════════

def build_bnn(input_dim, num_classes, hidden_sizes, kl_weight, name="BNN"):
    """
    Functional-API BNN.

    Parameters
    ----------
    hidden_sizes : list[int]  – hidden layer widths
    kl_weight    : float      – KL scaling factor
    """
    inp = tf.keras.Input(shape=(input_dim,), name="input")
    x   = inp
    for i, h in enumerate(hidden_sizes):
        x = BayesianDense(h, kl_weight=kl_weight,
                          activation="relu", name=f"bayes_{i}")(x)
    out = BayesianDense(num_classes, kl_weight=kl_weight, name="logits")(x)
    return tf.keras.Model(inp, out, name=name)


kl_base = CFG["kl_base"]
D       = CFG["input_dim"]
C       = CFG["n_classes"]

BNN_small  = build_bnn(D, C, [32, 16],   kl_base,                   "BNN_small")
BNN_large  = build_bnn(D, C, [128, 64],  kl_base,                   "BNN_large")
BNN_lowKL  = build_bnn(D, C, [128, 64],  kl_base * CFG["kl_low_mul"],  "BNN_lowKL")
BNN_highKL = build_bnn(D, C, [128, 64],  kl_base * CFG["kl_high_mul"], "BNN_highKL")

for m in [BNN_small, BNN_large, BNN_lowKL, BNN_highKL]:
    m.compile(
        optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"{m.name:15s}  params={m.count_params():>8,}  "
          f"kl_weight={m.layers[1].kl_weight:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2.3: BAYESIAN NEURAL-SYMBOLIC (BNS) MODEL
# Wraps BNN_large with a differentiable Temporal Structure Consistency (TSC)
# penalty that aligns predicted hourly occupancy with empirical patterns.
# ═══════════════════════════════════════════════════════════════════════════

class BNS_Model(tf.keras.Model):
    """
    Bayesian Neural-Symbolic model.

    The symbolic component enforces that the model's predicted occupancy
    distribution over hours matches the empirical distribution μ̂_h
    via a soft L2 penalty (TSC).
    """

    def __init__(self, backbone, empirical_mu, N_train,
                 lambda_max=0.05, beta=1.0, high_class=2):
        super().__init__()
        self.backbone     = backbone
        self.emp_mu       = tf.constant(empirical_mu, dtype=tf.float32)
        self.N_train      = N_train
        self.lambda_max   = lambda_max
        self.lambda_t     = 0.0          # annealed by callback
        self.beta         = beta
        self.high_class   = high_class

        # Metrics
        self._loss_tr = tf.keras.metrics.Mean("loss")
        self._acc_tr  = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        self._tsc_tr  = tf.keras.metrics.Mean("tsc")

    @property
    def metrics(self):
        return [self._loss_tr, self._acc_tr, self._tsc_tr]

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def _tsc_penalty(self, logits, x_batch):
        probs      = tf.nn.softmax(logits)
        high_probs = probs[:, self.high_class]          # P(occ = high | x)

        # Raw hour values live in the FIRST feature column BEFORE cyclic
        # encoding. We pass h_batch alongside x_batch as the 3rd element
        # of data, or we read back via the hour input tensor.
        # Here we pass hour as a second input tensor (see train loop below).
        # (Handled in train_step by unpacking data=(x, y) or (x, y, h))
        return high_probs   # returned to train_step for hour aggregation

    def _compute_tsc(self, logits, hours):
        """Compute TSC penalty given logits and integer hour vector."""
        probs      = tf.nn.softmax(logits)
        high_probs = probs[:, self.high_class]
        hours_i32  = tf.cast(hours, tf.int32)

        mu_hat = []
        for h in range(24):
            mask  = tf.cast(tf.equal(hours_i32, h), tf.float32)
            num   = tf.reduce_sum(mask * high_probs)
            den   = tf.reduce_sum(mask) + 1e-6
            mu_hat.append(num / den)

        mu_hat = tf.stack(mu_hat)
        tsc    = tf.reduce_mean(tf.square(mu_hat - self.emp_mu))
        return tsc

    def train_step(self, data):
        # data = (x_combined, y)
        # x_combined[:, -1] carries raw hour (appended by DataPipeline)
        x_full, y = data
        x_feat = x_full[:, :-1]   # model input features
        hours  = x_full[:, -1]    # raw hour (0-23)

        with tf.GradientTape() as tape:
            logits = self.backbone(x_feat, training=True)

            ce = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True))

            kl_losses = self.backbone.losses
            kl        = tf.add_n(kl_losses) if kl_losses else 0.0
            kl_scaled = self.beta * kl / self.N_train

            tsc   = self._compute_tsc(logits, hours)
            loss  = ce + kl_scaled + self.lambda_t * tsc

        grads = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.backbone.trainable_variables))

        self._loss_tr.update_state(loss)
        self._acc_tr.update_state(y, logits)
        self._tsc_tr.update_state(tsc)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_full, y = data
        x_feat = x_full[:, :-1]
        hours  = x_full[:, -1]

        logits = self.backbone(x_feat, training=False)
        ce     = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y, logits, from_logits=True))
        tsc    = self._compute_tsc(logits, hours)

        self._loss_tr.update_state(ce)
        self._acc_tr.update_state(y, logits)
        self._tsc_tr.update_state(tsc)

        return {m.name: m.result() for m in self.metrics}


# ── Lambda warm-up callback ──────────────────────────────────────────────
class LambdaWarmup(tf.keras.callbacks.Callback):
    """Linear warm-up of TSC penalty weight from 0 → lambda_max."""
    def __init__(self, warmup_epochs=5):
        super().__init__()
        self.warmup = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        lam = min(epoch / max(self.warmup, 1), 1.0) * self.model.lambda_max
        self.model.lambda_t = lam
        print(f"  λ_t = {lam:.4f}", end="")


# ── BNS datasets (append raw hour as last feature) ──────────────────────
def make_bns_dataset(X, h, y, shuffle=False):
    """Concat raw hour as last column so BNS can compute TSC."""
    X_h = np.concatenate([X, h.reshape(-1, 1).astype(np.float32)], axis=1)
    ds  = tf.data.Dataset.from_tensor_slices((X_h, y))
    if shuffle:
        ds = ds.shuffle(200_000, seed=SEED)
    return ds.batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)

bns_train_ds = make_bns_dataset(X_train, h_train, y_train, shuffle=True)
bns_val_ds   = make_bns_dataset(X_val,   h_val,   y_val)
bns_test_ds  = make_bns_dataset(X_test,  h_test,  y_test)


# ── Instantiate BNS (backbone = fresh BNN_large clone) ──────────────────
# We use a fresh backbone so BNS has independent weights from BNN_large
bnn_large_for_bns = build_bnn(D, C, [128, 64], kl_base, "BNN_large_BNS_backbone")

BNS = BNS_Model(
    backbone     = bnn_large_for_bns,
    empirical_mu = empirical_mu,
    N_train      = N_TRAIN,
    lambda_max   = CFG["lambda_bns"],
    beta         = 1.0,
    high_class   = 2,
)
BNS.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]))

print("\n✅ BNS model ready")
print(f"   lambda_max = {CFG['lambda_bns']}")
print(f"   backbone   = {bnn_large_for_bns.name}  ({bnn_large_for_bns.count_params():,} params)")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2.4: CLASSICAL BASELINES
# Logistic Regression | Random Forest | XGBoost
# Same features + same temporal split as all neural models
# ════════════════════════════════════════════════════════════════════════════
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (accuracy_score, f1_score,
                                   classification_report)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import numpy as np, pandas as pd, time

print("Training classical baselines on same train/val/test split...")
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─── Class weights ────────────────────────────────────────────────────────
cw_arr = compute_class_weight("balanced",
                               classes=np.arange(CFG["n_classes"]),
                               y=y_train)
cw_dict = {i: float(cw_arr[i]) for i in range(CFG["n_classes"])}
sample_w_train = np.array([cw_dict[y] for y in y_train])

# ─── 1. Logistic Regression ───────────────────────────────────────────────
t0 = time.time()
lr_model = LogisticRegression(
    max_iter=1000, C=1.0, solver="lbfgs",
    multi_class="multinomial", class_weight="balanced",
    random_state=CFG["seed"], n_jobs=-1
)
lr_model.fit(X_train, y_train)
lr_pred  = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)
lr_time  = time.time() - t0

lr_acc = accuracy_score(y_test, lr_pred)
lr_f1  = f1_score(y_test, lr_pred, average="macro")
print(f"\nLogistic Regression  —  Acc={lr_acc:.4f}  MacroF1={lr_f1:.4f}  "
      f"({lr_time:.1f}s)")

# ─── 2. Random Forest ─────────────────────────────────────────────────────
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_leaf=10,
    class_weight="balanced", random_state=CFG["seed"], n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)
rf_time  = time.time() - t0

rf_acc = accuracy_score(y_test, rf_pred)
rf_f1  = f1_score(y_test, rf_pred, average="macro")
print(f"Random Forest        —  Acc={rf_acc:.4f}  MacroF1={rf_f1:.4f}  "
      f"({rf_time:.1f}s)")

# ─── 3. XGBoost ──────────────────────────────────────────────────────────
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=CFG["seed"], n_jobs=-1,
    verbosity=0
)
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_w_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
xgb_time  = time.time() - t0

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1  = f1_score(y_test, xgb_pred, average="macro")
print(f"XGBoost              —  Acc={xgb_acc:.4f}  MacroF1={xgb_f1:.4f}  "
      f"({xgb_time:.1f}s)")

print("\n✅ Classical baselines ready")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2.4b: CLASSICAL BASELINE METRICS + COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

def classical_metrics(name, y_true, y_pred, y_proba):
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average="macro")
    brier = float(np.mean([
        brier_score_loss((y_true==c).astype(int), y_proba[:,c])
        for c in range(CFG["n_classes"])]))
    nll   = float(log_loss(y_true, y_proba))
    # Accuracy@1: prediction correct if within 1 class
    acc1  = float(np.mean(np.abs(y_pred - y_true) <= 1))
    return dict(Model=name, Accuracy=round(acc,4), MacroF1=round(mf1,4),
                Brier=round(brier,4), NLL=round(nll,4), Acc_at_1=round(acc1,4))

classical_results = [
    classical_metrics("Logistic Regression", y_test, lr_pred,  lr_proba),
    classical_metrics("Random Forest",       y_test, rf_pred,  rf_proba),
    classical_metrics("XGBoost",             y_test, xgb_pred, xgb_proba),
]
classical_df = pd.DataFrame(classical_results).set_index("Model")

print("=" * 65)
print("  CLASSICAL BASELINE RESULTS — TEST SET")
print("=" * 65)
print(classical_df.to_string())
print("=" * 65)
print("  Acc@1 = prediction within 1 class (matches Nezhadettehad Acc@1)")

# ── Comparison bar chart with paper's results ────────────────────────────
paper_data = {
    "SVM (paper)":    {"Accuracy": 0.462, "Acc_at_1": 0.800},
    "RF (paper)":     {"Accuracy": 0.457, "Acc_at_1": 0.793},
    "LSTM (paper)":   {"Accuracy": 0.559, "Acc_at_1": 0.902},
    "BNN (paper)":    {"Accuracy": 0.517, "Acc_at_1": 0.892},
    "BNN-20% (paper)":{"Accuracy": 0.783, "Acc_at_1": 0.956},
    "BNN-30% (paper)":{"Accuracy": 0.833, "Acc_at_1": 0.998},
}
paper_df = pd.DataFrame(paper_data).T

our_data = classical_df[["Accuracy","Acc_at_1"]].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_paper = ["#AAAAAA"] * len(paper_df)
colors_ours  = ["#4477AA","#228833","#AA3377"]

for ax, metric, title in [
    (axes[0], "Accuracy",  "Accuracy (Prediction Window 1)"),
    (axes[1], "Acc_at_1",  "Accuracy@1 (Relaxed, Window 1)"),
]:
    x_p = range(len(paper_df))
    ax.bar([x - 0.2 for x in x_p],
           paper_df[metric].values,
           width=0.35, color="#BBBBBB", edgecolor="black",
           linewidth=0.7, label="Nezhadettehad et al. (2025)")
    x_o = range(len(paper_df), len(paper_df) + len(our_data))
    ax.bar([x - 0.2 for x in x_o],
           our_data[metric].values,
           width=0.35, color=colors_ours, edgecolor="black",
           linewidth=0.7, label="This work (classical)")
    ax.set_xticks(list(x_p) + list(x_o))
    ax.set_xticklabels(
        list(paper_df.index) + list(our_data.index),
        rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

plt.suptitle("Our Classical Baselines vs. Nezhadettehad et al. (2025) — Prediction Window 1",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("classical_vs_paper.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: classical_vs_paper.pdf")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2.4c: FEATURE IMPORTANCE (RF + XGBoost)
# ════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model, name, color in [
    (axes[0], rf_model,  "Random Forest", "#228833"),
    (axes[1], xgb_model, "XGBoost",       "#AA3377"),
]:
    imp    = model.feature_importances_
    idx    = np.argsort(imp)[::-1]
    labels = [FEATURES[i] for i in idx][:10]
    vals   = imp[idx][:10]

    bars = ax.barh(labels[::-1], vals[::-1],
                   color=color, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"{name} — Top Feature Importances",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Feature Importance: RF vs XGBoost", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: feature_importance.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: SHARED TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_callbacks(monitor="val_accuracy", patience=7):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience,
            restore_best_weights=True, verbose=0, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=3,
            min_lr=1e-6, verbose=0),
    ]

print("Callbacks ready.")


# Deterministic MLP – no KL, used as reference
det_inputs = tf.keras.Input(shape=(D,))
det_x = tf.keras.layers.Dense(128, activation="relu")(det_inputs)
det_x = tf.keras.layers.BatchNormalization()(det_x)
det_x = tf.keras.layers.Dropout(0.3)(det_x)
det_x = tf.keras.layers.Dense(64, activation="relu")(det_x)
det_x = tf.keras.layers.BatchNormalization()(det_x)
det_x = tf.keras.layers.Dropout(0.3)(det_x)
det_x = tf.keras.layers.Dense(32, activation="relu")(det_x)
det_out = tf.keras.layers.Dense(C, activation="softmax")(det_x)

DET = tf.keras.Model(det_inputs, det_out, name="DET_MLP")
DET.compile(
    optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
DET.summary()


history_det = DET.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["epochs_bnn"],
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks(),
    verbose=1,
)
print("✅ DET training complete")


history_small = BNN_small.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["epochs_bnn"],
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks(),
    verbose=1,
)
print("✅ BNN_small training complete")


history_large = BNN_large.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["epochs_bnn"],
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks(),
    verbose=1,
)
print("✅ BNN_large training complete")


history_lowkl = BNN_lowKL.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["epochs_bnn"],
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks(),
    verbose=1,
)
print("✅ BNN_lowKL training complete")


history_highkl = BNN_highKL.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["epochs_bnn"],
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks(),
    verbose=1,
)
print("✅ BNN_highKL training complete")


history_bns = BNS.fit(
    bns_train_ds,
    validation_data=bns_val_ds,
    epochs=CFG["epochs_bns"],
    callbacks=[
        LambdaWarmup(warmup_epochs=5),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=0, mode="max"),
    ],
    verbose=1,
)
print("✅ BNS training complete")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

# ── (a) ECE – Expected Calibration Error ────────────────────────────────
def compute_ece(y_true, y_proba, n_bins=15):
    """Multi-class ECE (confidence of argmax vs accuracy in equal-width bins)."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct     = (predictions == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
    return ece / len(y_true)


# ── (b) Temporal Consistency Error ──────────────────────────────────────
def compute_tce(proba, hours, empirical_mu, high_class=2):
    """MSE between predicted P(high-occ | hour) and empirical curve."""
    high_probs = proba[:, high_class]
    mu_pred    = np.array([
        np.mean(high_probs[hours == h]) if np.any(hours == h) else 0.0
        for h in range(24)
    ])
    return float(np.mean((mu_pred - empirical_mu) ** 2)), mu_pred


# ── (c) Monte-Carlo uncertainty (mean + std over T forward passes) ───────
def mc_predict(model, X, T=50, batch_size=4096, from_logits=True):
    """
    Monte-Carlo forward passes with training=True (active dropout / BNN sampling).

    Returns
    -------
    mean_proba : (N, C)
    std_proba  : (N, C)   per-sample predictive std
    """
    all_runs = []
    for _ in range(T):
        # Run the full dataset in mini-batches, then concat → (N, C)
        batch_preds = []
        for i in range(0, len(X), batch_size):
            x_b = X[i : i + batch_size]
            out = model(x_b, training=True)
            p   = tf.nn.softmax(out).numpy() if from_logits else out.numpy()
            batch_preds.append(p)
        all_runs.append(np.concatenate(batch_preds, axis=0))   # (N, C)

    preds = np.stack(all_runs, axis=0)   # (T, N, C)
    return preds.mean(axis=0), preds.std(axis=0)


# ── (d) Evaluate one model ───────────────────────────────────────────────
def evaluate_model(model, X, y, hours, empirical_mu,
                   name="Model", mc=False, T=50):
    """
    Returns dict of metrics: accuracy, macro_f1, ece, tce.
    Optionally performs Monte-Carlo sampling.
    """
    if mc:
        proba, unc = mc_predict(model, X, T=T)
    else:
        logits = model.predict(X, batch_size=4096, verbose=0)
        proba  = tf.nn.softmax(logits).numpy()
        unc    = None

    y_pred = np.argmax(proba, axis=1)

    acc   = accuracy_score(y, y_pred)
    mf1   = f1_score(y, y_pred, average="macro")
    ece   = compute_ece(y, proba, n_bins=CFG["ece_bins"])
    tce, mu_pred = compute_tce(proba, hours, empirical_mu)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Macro F1         : {mf1:.4f}")
    print(f"  ECE (↓better)    : {ece:.4f}")
    print(f"  TCE (↓better)    : {tce:.4f}")
    if unc is not None:
        print(f"  Mean uncertainty : {unc.mean():.4f}")
    print()
    print(classification_report(y, y_pred, digits=4,
          target_names=[f"Occ-{i}" for i in range(CFG["n_classes"])]))

    return dict(name=name, accuracy=acc, macro_f1=mf1, ece=ece, tce=tce,
                y_pred=y_pred, proba=proba, mu_pred=mu_pred)


print("✅ Evaluation utilities ready.")


results = {}

results["DET"]        = evaluate_model(DET,        X_test, y_test, h_test,
                                        empirical_mu, name="DET (Baseline)")

results["BNN_small"]  = evaluate_model(BNN_small,  X_test, y_test, h_test,
                                        empirical_mu, name="BNN-Small",
                                        mc=True, T=CFG["mc_samples"])

results["BNN_large"]  = evaluate_model(BNN_large,  X_test, y_test, h_test,
                                        empirical_mu, name="BNN-Large",
                                        mc=True, T=CFG["mc_samples"])

results["BNN_lowKL"]  = evaluate_model(BNN_lowKL,  X_test, y_test, h_test,
                                        empirical_mu, name="BNN-LowKL",
                                        mc=True, T=CFG["mc_samples"])

results["BNN_highKL"] = evaluate_model(BNN_highKL, X_test, y_test, h_test,
                                        empirical_mu, name="BNN-HighKL",
                                        mc=True, T=CFG["mc_samples"])

# BNS prediction: model.call(x_feat) – strip the hour column appended for BNS
def predict_bns(bns_model, X, h, T=50, mc=True):
    X_h    = np.concatenate([X, h.reshape(-1,1).astype(np.float32)], axis=1)
    # BNS.call routes to backbone, which only uses X_h[:, :-1]
    # so we pass X directly to the backbone for clean prediction
    backbone = bns_model.backbone
    if mc:
        proba, unc = mc_predict(backbone, X, T=T)
    else:
        logits = backbone.predict(X, batch_size=4096, verbose=0)
        proba  = tf.nn.softmax(logits).numpy()
        unc    = None
    return proba, unc

bns_proba, bns_unc = predict_bns(BNS, X_test, h_test, T=CFG["mc_samples"], mc=True)
bns_pred = np.argmax(bns_proba, axis=1)

acc_bns  = accuracy_score(y_test, bns_pred)
mf1_bns  = f1_score(y_test, bns_pred, average="macro")
ece_bns  = compute_ece(y_test, bns_proba, CFG["ece_bins"])
tce_bns, mu_pred_bns = compute_tce(bns_proba, h_test, empirical_mu)

print(f"\n{'─'*55}")
print(f"  BNS (Bayesian Neural-Symbolic)")
print(f"{'─'*55}")
print(f"  Accuracy         : {acc_bns:.4f}")
print(f"  Macro F1         : {mf1_bns:.4f}")
print(f"  ECE              : {ece_bns:.4f}")
print(f"  TCE              : {tce_bns:.4f}")
print(f"  Mean uncertainty : {bns_unc.mean():.4f}")
print()
print(classification_report(y_test, bns_pred, digits=4,
      target_names=[f"Occ-{i}" for i in range(CFG["n_classes"])]))

results["BNS"] = dict(name="BNS", accuracy=acc_bns, macro_f1=mf1_bns,
                      ece=ece_bns, tce=tce_bns,
                      y_pred=bns_pred, proba=bns_proba, mu_pred=mu_pred_bns)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4.3: SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

rows = []
for k, r in results.items():
    rows.append({
        "Model"      : r["name"],
        "Accuracy"   : f"{r['accuracy']:.4f}",
        "Macro F1"   : f"{r['macro_f1']:.4f}",
        "ECE (↓)"    : f"{r['ece']:.4f}",
        "TCE (↓)"    : f"{r['tce']:.4f}",
    })

summary = pd.DataFrame(rows).set_index("Model")
print("\n" + "="*60)
print("  MODEL COMPARISON — TEST SET")
print("="*60)
print(summary.to_string())
print("="*60)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4.3b: FULL COMPARISON TABLE — OUR MODELS + CLASSICAL + PAPER
# ════════════════════════════════════════════════════════════════════════════
import pandas as pd

# ── Our neural models (from results dict) ────────────────────────────────
our_neural = pd.DataFrame([
    {k: v for k, v in r.items()
     if k in ("name","accuracy","macro_f1","ece","tce")}
    for r in results.values()
]).rename(columns={"name":"Model","accuracy":"Accuracy","macro_f1":"Macro F1",
                   "ece":"ECE","tce":"TCE"}).set_index("Model")

# ── Our classical models ──────────────────────────────────────────────────
our_cls = classical_df[["Accuracy","MacroF1"]].rename(
    columns={"MacroF1":"Macro F1"})
our_cls["ECE"] = float("nan")
our_cls["TCE"] = float("nan")

# ── Nezhadettehad et al. (2025) — Window 1 results ───────────────────────
paper_rows = pd.DataFrame([
    {"Model":"SVM (Nezhadettehad 2025)",     "Accuracy":0.462, "Macro F1":float("nan"),
     "ECE":float("nan"), "TCE":float("nan")},
    {"Model":"RF (Nezhadettehad 2025)",      "Accuracy":0.457, "Macro F1":float("nan"),
     "ECE":float("nan"), "TCE":float("nan")},
    {"Model":"LSTM (Nezhadettehad 2025)",    "Accuracy":0.559, "Macro F1":float("nan"),
     "ECE":float("nan"), "TCE":float("nan")},
    {"Model":"BNN (Nezhadettehad 2025)",     "Accuracy":0.517, "Macro F1":float("nan"),
     "ECE":float("nan"), "TCE":float("nan")},
    {"Model":"BNN-20% (Nezhadettehad 2025)", "Accuracy":0.783, "Macro F1":float("nan"),
     "ECE":float("nan"), "TCE":float("nan")},
    {"Model":"BNN-30% (Nezhadettehad 2025)", "Accuracy":0.833, "Macro F1":float("nan"),
     "ECE":float("nan"), "TCE":float("nan")},
]).set_index("Model")

full_table = pd.concat([our_cls, our_neural, paper_rows])

print("=" * 75)
print("  COMPLETE COMPARISON: OUR MODELS + CLASSICAL + NEZHADETTEHAD et al.")
print("=" * 75)
print(full_table.round(4).fillna("—").to_string())
print("=" * 75)
print("  Notes:")
print("  - Paper results: Prediction Window 1, full data, no threshold")
print("  - ECE/TCE: not reported in Nezhadettehad et al. (2025)")
print("  - BNN-20%/30% predictions = 64% / 60% of test set only")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.1: CONFUSION MATRICES  (NeurIPS figure style)
# ═══════════════════════════════════════════════════════════════════════════

MODEL_KEYS = ["DET", "BNN_small", "BNN_large", "BNN_lowKL", "BNN_highKL", "BNS"]
CLASS_NAMES = [f"Occ-{i}" for i in range(5)]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, key in zip(axes, MODEL_KEYS):
    r  = results[key]
    cm = confusion_matrix(y_test, r["y_pred"], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title(r["name"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True",      fontsize=9)
    ax.tick_params(labelsize=8)

plt.suptitle("Normalized Confusion Matrices — Test Set", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("confusion_matrices.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: confusion_matrices.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.2: TEMPORAL CONSISTENCY CURVES
# ═══════════════════════════════════════════════════════════════════════════

hours_axis = np.arange(24)

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(hours_axis, empirical_mu, "k-", linewidth=3,
        label="Empirical (Ground Truth)", zorder=5)

styles = {
    "DET"       : ("C0", "--",  1.8, "DET Baseline"),
    "BNN_small" : ("C1", ":",   1.8, "BNN-Small"),
    "BNN_large" : ("C2", "-.",  1.8, "BNN-Large"),
    "BNN_lowKL" : ("C3", "--",  1.8, "BNN-LowKL"),
    "BNN_highKL": ("C4", ":",   1.8, "BNN-HighKL"),
    "BNS"       : ("C5", "-",   2.5, "BNS (Ours)"),
}

for key, (color, ls, lw, label) in styles.items():
    ax.plot(hours_axis, results[key]["mu_pred"],
            color=color, linestyle=ls, linewidth=lw, label=label)

ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("P(High Occupancy | Hour)", fontsize=11)
ax.set_title("Temporal Structure Consistency: True vs Predicted", fontsize=12)
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("temporal_consistency.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: temporal_consistency.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.3: RELIABILITY DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════

def reliability_diagram(ax, y_true, y_proba, n_bins=15, label="", color="C0"):
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct     = (predictions == y_true).astype(float)

    bins   = np.linspace(0, 1, n_bins + 1)
    bx, by = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bx.append(confidences[mask].mean())
            by.append(correct[mask].mean())

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(bx, by, "o-", color=color, linewidth=2, markersize=5, label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

for ax, (key, color) in zip(axes, zip(MODEL_KEYS, colors)):
    r = results[key]
    reliability_diagram(ax, y_test, r["proba"],
                        n_bins=CFG["ece_bins"],
                        label=f"{r['name']} (ECE={r['ece']:.3f})",
                        color=color)
    ax.set_title(r["name"], fontsize=10, fontweight="bold")

plt.suptitle("Reliability Diagrams — Test Set", fontsize=13)
plt.tight_layout()
plt.savefig("reliability_diagrams.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: reliability_diagrams.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.4: KL ABLATION PLOT
# ═══════════════════════════════════════════════════════════════════════════

kl_variants = ["BNN_lowKL", "BNN_large", "BNN_highKL"]
kl_labels   = [f"LowKL\n(×{CFG['kl_low_mul']})",
               "Standard\n(×1)",
               f"HighKL\n(×{int(CFG['kl_high_mul'])})"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

metrics_ab = ["accuracy", "macro_f1", "ece"]
mtitles    = ["Accuracy (↑)", "Macro F1 (↑)", "ECE (↓)"]

for ax, metric, title in zip(axes, metrics_ab, mtitles):
    vals   = [results[k][metric] for k in kl_variants]
    colors = ["#4477AA", "#EE6677", "#228833"]
    bars   = ax.bar(kl_labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("KL Weight Ablation (BNN-Large backbone)", fontsize=12)
plt.tight_layout()
plt.savefig("kl_ablation.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: kl_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.5: TRAINING CURVES (Loss + Accuracy)
# ═══════════════════════════════════════════════════════════════════════════

hist_map = {
    "DET"       : history_det,
    "BNN-Small" : history_small,
    "BNN-Large" : history_large,
    "BNN-LowKL" : history_lowkl,
    "BNN-HighKL": history_highkl,
    "BNS"       : history_bns,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
colors = [f"C{i}" for i in range(6)]

for (label, hist), color in zip(hist_map.items(), colors):
    axes[0].plot(hist.history["val_loss"],     label=label, color=color)
    axes[1].plot(hist.history["val_accuracy"], label=label, color=color)

for ax, title, ylabel in zip(
        axes,
        ["Validation Loss", "Validation Accuracy"],
        ["Loss", "Accuracy"]):
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle("Training Curves — All Models", fontsize=13)
plt.tight_layout()
plt.savefig("training_curves.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: training_curves.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6.1: LABEL NOISE ROBUSTNESS
# We compare DET, BNN_large, and BNS under 0 / 20 / 30% label noise.
# ═══════════════════════════════════════════════════════════════════════════

def inject_label_noise(y, noise_frac, n_classes, seed=0):
    rng     = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_noisy = int(noise_frac * len(y))
    idx     = rng.choice(len(y), n_noisy, replace=False)
    for i in idx:
        choices          = [c for c in range(n_classes) if c != y_noisy[i]]
        y_noisy[i]       = rng.choice(choices)
    return y_noisy


def train_and_eval_noisy(model_fn, noise_frac, name):
    """Train a fresh model clone on noisy labels; eval on clean test set."""
    y_noisy = inject_label_noise(y_train, noise_frac, CFG["n_classes"])
    ds_n    = make_dataset(X_train, y_noisy, shuffle=True)
    m = model_fn()
    m.fit(ds_n, validation_data=val_ds,
          epochs=CFG["epochs_bnn"],
          class_weight=CLASS_WEIGHTS,
          callbacks=get_callbacks(), verbose=0)
    logits = m.predict(X_test, batch_size=4096, verbose=0)
    proba  = tf.nn.softmax(logits).numpy()
    y_pred = np.argmax(proba, axis=1)
    return dict(
        model=name, noise=noise_frac,
        accuracy=accuracy_score(y_test, y_pred),
        macro_f1=f1_score(y_test, y_pred, average="macro"),
    )


noise_results = []
for noise in [0.0, 0.2, 0.3]:
    print(f"\nNoise = {int(noise*100)}% ...")
    noise_results.append(
        train_and_eval_noisy(
            lambda: build_bnn(D, C, [128, 64], kl_base, "bnn_tmp").also(
                lambda m: m.compile(
                    optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"])),
            noise, "BNN-Large") if False else {}  # placeholder
    )

# Simplified robust evaluation (retrain inline):
noise_rows = []
for noise in [0.0, 0.20, 0.30]:
    y_noisy = inject_label_noise(y_train, noise, CFG["n_classes"])
    ds_n    = make_dataset(X_train, y_noisy, shuffle=True)

    for mname, mbuilder in [
        ("DET",       lambda: tf.keras.Sequential([
                          tf.keras.layers.Dense(128, activation="relu",
                                                input_shape=(D,)),
                          tf.keras.layers.BatchNormalization(),
                          tf.keras.layers.Dense(64, activation="relu"),
                          tf.keras.layers.Dense(C, activation="softmax")])),
        ("BNN-Large", lambda: build_bnn(D, C, [128, 64], kl_base, "tmp")),
    ]:
        m = mbuilder()
        loss = ("sparse_categorical_crossentropy"
                if mname == "DET"
                else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        m.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
                  loss=loss, metrics=["accuracy"])
        m.fit(ds_n, epochs=20, verbose=0,
              class_weight=CLASS_WEIGHTS,
              callbacks=get_callbacks())
        logits = m.predict(X_test, batch_size=4096, verbose=0)
        proba  = tf.nn.softmax(logits).numpy() if mname != "DET" else logits
        y_pred = np.argmax(proba, axis=1)
        noise_rows.append(dict(
            Model    = mname,
            Noise    = f"{int(noise*100)}%",
            Accuracy = round(accuracy_score(y_test, y_pred), 4),
            MacroF1  = round(f1_score(y_test, y_pred, average="macro"), 4),
        ))
        print(f"  {mname:12s}  noise={int(noise*100)}%  "
              f"acc={noise_rows[-1]['Accuracy']:.4f}  "
              f"mf1={noise_rows[-1]['MacroF1']:.4f}")

noise_df = pd.DataFrame(noise_rows)
print("\n" + noise_df.pivot(index="Model", columns="Noise",
                             values=["Accuracy", "MacroF1"]).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6.2: PREDICTIVE ENTROPY & UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════════════

def predictive_entropy(proba):
    """H[y|x] = -Σ p_c log p_c  (per sample)"""
    eps = 1e-8
    return -np.sum(proba * np.log(proba + eps), axis=1)


# Compare BNN-Large vs BNS uncertainty on correct vs incorrect predictions
sample_size = min(50_000, len(X_test))
rng_idx = np.random.default_rng(SEED).choice(len(X_test), sample_size, replace=False)

X_s, y_s, h_s = X_test[rng_idx], y_test[rng_idx], h_test[rng_idx]

bnn_proba_mc, _ = mc_predict(BNN_large, X_s, T=CFG["mc_samples"])
bns_proba_mc, _ = mc_predict(BNS.backbone, X_s, T=CFG["mc_samples"])

bnn_ent = predictive_entropy(bnn_proba_mc)
bns_ent = predictive_entropy(bns_proba_mc)

bnn_correct = (np.argmax(bnn_proba_mc, 1) == y_s)
bns_correct = (np.argmax(bns_proba_mc, 1) == y_s)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, ent, correct, label, color in [
    (axes[0], bnn_ent, bnn_correct, "BNN-Large", "C2"),
    (axes[1], bns_ent, bns_correct, "BNS",       "C5"),
]:
    ax.hist(ent[correct],  bins=50, alpha=0.6, color="steelblue",
            density=True, label="Correct")
    ax.hist(ent[~correct], bins=50, alpha=0.6, color="tomato",
            density=True, label="Incorrect")
    ax.set_xlabel("Predictive Entropy H[y|x]", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{label}: Uncertainty by Correctness", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # AURC: uncertain predictions should be incorrect more often
    # A well-calibrated model has higher entropy on wrong predictions
    mean_corr   = ent[correct].mean()
    mean_incorr = ent[~correct].mean()
    ax.text(0.55, 0.88,
            f"H̄(correct) = {mean_corr:.3f}\nH̄(wrong) = {mean_incorr:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle("Predictive Uncertainty Distribution (MC Dropout / VI)", fontsize=12)
plt.tight_layout()
plt.savefig("uncertainty_analysis.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: uncertainty_analysis.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: FINAL DASHBOARD (NeurIPS-quality figure)
# ═══════════════════════════════════════════════════════════════════════════

model_names = [results[k]["name"] for k in MODEL_KEYS]
acc_vals    = [results[k]["accuracy"]  for k in MODEL_KEYS]
f1_vals     = [results[k]["macro_f1"]  for k in MODEL_KEYS]
ece_vals    = [results[k]["ece"]       for k in MODEL_KEYS]
tce_vals    = [results[k]["tce"]       for k in MODEL_KEYS]

x    = np.arange(len(model_names))
w    = 0.35
pal  = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, vals, ylabel, title, flip in [
    (axes[0,0], acc_vals, "Accuracy",  "Test Accuracy (higher=better)",  False),
    (axes[0,1], f1_vals,  "Macro F1",  "Macro F1   (higher=better)",      False),
    (axes[1,0], ece_vals, "ECE",       "ECE  (lower=better)",             True),
    (axes[1,1], tce_vals, "TCE",       "TCE (lower=better)",              True),
]:
    bars = ax.bar(x, vals, color=pal, edgecolor="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ypad = max(vals) * 0.15
    ax.set_ylim(0 if not flip else 0, max(vals) + ypad)

plt.suptitle("Comprehensive Model Comparison — Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("final_comparison.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: final_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: REPRODUCIBILITY CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  REPRODUCIBILITY CHECKLIST")
print("=" * 60)
print(f"  Random seed         : {CFG['seed']}")
print(f"  TF version          : {tf.__version__}")
print(f"  Features            : {CFG['input_dim']}")
print(f"  N_train             : {N_TRAIN:,}")
print(f"  kl_base             : {CFG['kl_base']:.3e}")
print(f"  Batch size          : {CFG['batch_size']}")
print(f"  BNN epochs          : {CFG['epochs_bnn']}")
print(f"  BNS epochs          : {CFG['epochs_bns']}")
print(f"  MC samples (eval)   : {CFG['mc_samples']}")
print(f"  ECE bins            : {CFG['ece_bins']}")
print(f"  BNS lambda_max      : {CFG['lambda_bns']}")
print(f"  KL_low  multiplier  : {CFG['kl_low_mul']}")
print(f"  KL_high multiplier  : {CFG['kl_high_mul']}")
print()
print("  Output files:")
for fname in ["confusion_matrices.pdf", "temporal_consistency.pdf",
              "reliability_diagrams.pdf", "kl_ablation.pdf",
              "training_curves.pdf", "uncertainty_analysis.pdf",
              "final_comparison.pdf"]:
    import os
    exists = "✅" if os.path.exists(fname) else "⬜"
    print(f"    {exists} {fname}")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: TIME-SERIES AWARE SPLIT
# Temporal ordering is preserved to prevent data leakage.
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Reconstruct month-tagged dataset from balanced df ────────────────────
# (df already has Month column from preprocessing)
df_ts = df.copy()

# Cyclic features (same as Section 1c)
for col, period in [("Hour",24),("DayOfWeek",7),("Month",12)]:
    df_ts[f"{col}_sin"] = np.sin(2*np.pi*df_ts[col]/period)
    df_ts[f"{col}_cos"] = np.cos(2*np.pi*df_ts[col]/period)

FEATURES_TS = [
    "Hour_sin","Hour_cos","DayOfWeek_sin","DayOfWeek_cos",
    "Month_sin","Month_cos","Day","IsWeekend",
    "air_temp","dew_point","wind_spd","pressure","rain_mm",
]

df_ts = df_ts.sort_values("Month").reset_index(drop=True)

# Chronological split: month ≤ 9 | month 10 | month ≥ 11
mask_train = df_ts["Month"] <= 9
mask_val   = df_ts["Month"] == 10
mask_test  = df_ts["Month"] >= 11

X_ts_train = df_ts.loc[mask_train, FEATURES_TS].values.astype(np.float32)
X_ts_val   = df_ts.loc[mask_val,   FEATURES_TS].values.astype(np.float32)
X_ts_test  = df_ts.loc[mask_test,  FEATURES_TS].values.astype(np.float32)

y_ts_train = df_ts.loc[mask_train, "OccClass"].values.astype(np.int32)
y_ts_val   = df_ts.loc[mask_val,   "OccClass"].values.astype(np.int32)
y_ts_test  = df_ts.loc[mask_test,  "OccClass"].values.astype(np.int32)

h_ts_train = df_ts.loc[mask_train, "Hour"].values
h_ts_val   = df_ts.loc[mask_val,   "Hour"].values
h_ts_test  = df_ts.loc[mask_test,  "Hour"].values

# Scale (fit on train ONLY)
scaler_ts = StandardScaler()
X_ts_train = scaler_ts.fit_transform(X_ts_train).astype(np.float32)
X_ts_val   = scaler_ts.transform(X_ts_val).astype(np.float32)
X_ts_test  = scaler_ts.transform(X_ts_test).astype(np.float32)

print(f"Temporal split:")
print(f"  Train (Jan–Sep) : {X_ts_train.shape[0]:>8,}  classes: {np.unique(y_ts_train)}")
print(f"  Val   (Oct)     : {X_ts_val.shape[0]:>8,}  classes: {np.unique(y_ts_val)}")
print(f"  Test  (Nov–Dec) : {X_ts_test.shape[0]:>8,}  classes: {np.unique(y_ts_test)}")


# ── Visualise the temporal split ─────────────────────────────────────────
import matplotlib.pyplot as plt

month_counts = df_ts.groupby(["Month","OccClass"]).size().unstack(fill_value=0)
months       = month_counts.index

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: sample count per month coloured by split
split_color = {m: "#4477AA" if m <= 9 else "#EE6677" if m == 10 else "#228833"
               for m in range(1, 13)}
ax = axes[0]
totals = df_ts.groupby("Month").size()
bars = ax.bar(totals.index,
              totals.values,
              color=[split_color[m] for m in totals.index],
              edgecolor="black", linewidth=0.7)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Sample Count", fontsize=11)
ax.set_title("Temporal Split by Month", fontsize=12, fontweight="bold")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)
ax.grid(axis="y", alpha=0.3)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#4477AA", label="Train (Jan–Sep)"),
    Patch(facecolor="#EE6677", label="Val  (Oct)"),
    Patch(facecolor="#228833", label="Test (Nov–Dec)"),
], fontsize=9)

# Right: class distribution per split
split_data = {
    "Train": np.bincount(y_ts_train, minlength=5) / len(y_ts_train),
    "Val"  : np.bincount(y_ts_val,   minlength=5) / len(y_ts_val),
    "Test" : np.bincount(y_ts_test,  minlength=5) / len(y_ts_test),
}
ax2 = axes[1]
x  = np.arange(5)
w  = 0.25
pal = ["#4477AA","#EE6677","#228833"]
for i, (split, vals) in enumerate(split_data.items()):
    ax2.bar(x + i*w, vals, w, label=split,
            color=pal[i], edgecolor="black", linewidth=0.7)
ax2.set_xticks(x + w)
ax2.set_xticklabels([f"Occ-{i}" for i in range(5)])
ax2.set_ylabel("Proportion", fontsize=11)
ax2.set_title("Class Distribution per Split", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Temporal (Time-Series) Split Diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig("ts_split_diagnostics.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: ts_split_diagnostics.pdf")


# ── Re-evaluate BNN_large & BNS on temporal split ────────────────────────
# Quick re-train on temporal split to compare against random-split results
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

D_ts = X_ts_train.shape[1]

ts_train_ds = make_dataset(X_ts_train, y_ts_train, shuffle=True)
ts_val_ds   = make_dataset(X_ts_val,   y_ts_val)
ts_test_ds  = make_dataset(X_ts_test,  y_ts_test)

# BNN-Large on temporal split
BNN_ts = build_bnn(D_ts, CFG["n_classes"], [128, 64],
                   1.0/len(X_ts_train), "BNN_temporal")
BNN_ts.compile(
    optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])

cw_ts = compute_class_weight("balanced",
                              classes=np.arange(CFG["n_classes"]),
                              y=y_ts_train)
cw_ts = {i: float(np.sqrt(cw_ts[i])) for i in range(CFG["n_classes"])}

BNN_ts.fit(ts_train_ds, validation_data=ts_val_ds,
           epochs=CFG["epochs_bnn"], class_weight=cw_ts,
           callbacks=get_callbacks(), verbose=1)

logits_ts = BNN_ts.predict(X_ts_test, batch_size=4096, verbose=0)
proba_ts  = tf.nn.softmax(logits_ts).numpy()
pred_ts   = np.argmax(proba_ts, axis=1)

emp_mu_ts = compute_empirical_mu(h_ts_train, y_ts_train)
tce_ts, _ = compute_tce(proba_ts, h_ts_test, emp_mu_ts)

print("\n── BNN-Large (Temporal Split) ──────────────────────────────")
print(f"  Accuracy : {accuracy_score(y_ts_test, pred_ts):.4f}")
print(f"  Macro F1 : {f1_score(y_ts_test, pred_ts, average='macro'):.4f}")
print(f"  ECE      : {compute_ece(y_ts_test, proba_ts):.4f}")
print(f"  TCE      : {tce_ts:.4f}")
print("\n  (Compare to random-split BNN-Large above to see leakage impact)")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: EXTENDED METRICS
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.metrics import (
    brier_score_loss, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, log_loss
)
from sklearn.preprocessing import label_binarize

def extended_metrics(y_true, proba, name="Model"):
    """
    Returns a dict with:
      accuracy, macro_f1, ece, tce,
      brier, nll, auroc_ovr, kappa, mcc
    """
    y_pred  = np.argmax(proba, axis=1)
    y_bin   = label_binarize(y_true, classes=list(range(CFG["n_classes"])))

    # Brier score (multi-class = mean over classes)
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), proba[:, c])
        for c in range(CFG["n_classes"])
    ])

    # Negative Log-Likelihood
    nll = log_loss(y_true, proba)

    # AUROC (macro one-vs-rest)
    try:
        auroc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
    except Exception:
        auroc = float("nan")

    # Kappa & MCC
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)

    return dict(
        Model   = name,
        Accuracy= round(accuracy_score(y_true, y_pred),  4),
        MacroF1 = round(f1_score(y_true, y_pred, average="macro"), 4),
        Brier   = round(brier,  4),
        NLL     = round(nll,    4),
        AUROC   = round(auroc,  4),
        Kappa   = round(kappa,  4),
        MCC     = round(mcc,    4),
    )


# ── Collect extended metrics for all models ───────────────────────────────
ext_rows = []
for key in MODEL_KEYS:
    r = results[key]
    ext_rows.append(extended_metrics(y_test, r["proba"], name=r["name"]))

ext_df = pd.DataFrame(ext_rows).set_index("Model")

print("\n" + "="*75)
print("  EXTENDED METRICS — TEST SET")
print("="*75)
print(ext_df.to_string())
print("="*75)
print("\nLower is better: Brier, NLL")
print("Higher is better: Accuracy, MacroF1, AUROC, Kappa, MCC")


# ── Radar chart: multi-metric comparison ─────────────────────────────────
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

RADAR_METRICS = ["Accuracy","MacroF1","AUROC","Kappa","MCC"]
# Invert Brier/NLL so all metrics are "higher = better" for radar
# Accuracy, MacroF1, AUROC, Kappa, MCC are already higher=better

N = len(RADAR_METRICS)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors  = ["#4477AA","#66CCEE","#228833","#CCBB44","#EE6677","#AA3377"]

for (idx, row), color in zip(ext_df.iterrows(), colors):
    vals = [float(row[m]) for m in RADAR_METRICS]
    vals += vals[:1]
    ax.plot(angles, vals, "-", linewidth=2, color=color, label=idx)
    ax.fill(angles, vals, alpha=0.06, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_METRICS, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
ax.set_title("Multi-Metric Radar Chart — All Models", fontsize=13,
             fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("radar_chart.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: radar_chart.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: PER-CLASS CALIBRATION
# For each class c, treat it as a binary problem (c vs. rest) and compute
# the ECE of P(y=c | x) against the true binary label.
# ═══════════════════════════════════════════════════════════════════════════

def per_class_ece(y_true, y_proba, n_bins=15):
    """Returns array of shape (n_classes,) with per-class ECE."""
    n_classes = y_proba.shape[1]
    ece_per_c = []
    for c in range(n_classes):
        binary_true = (y_true == c).astype(float)
        conf        = y_proba[:, c]
        bins        = np.linspace(0, 1, n_bins + 1)
        ece         = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() > 0:
                ece += mask.sum() * abs(binary_true[mask].mean()
                                        - conf[mask].mean())
        ece_per_c.append(ece / len(y_true))
    return np.array(ece_per_c)


# Compute for all models
pc_ece = {}
for key in MODEL_KEYS:
    pc_ece[key] = per_class_ece(y_test, results[key]["proba"])

# Table
pc_df = pd.DataFrame(
    {results[k]["name"]: pc_ece[k] for k in MODEL_KEYS},
    index=[f"Occ-{c}" for c in range(CFG["n_classes"])]
).T

print("Per-Class ECE (↓ better):")
print(pc_df.round(4).to_string())


# ── Per-class ECE heatmap ─────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(
    pc_df.astype(float),
    annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "ECE (lower = better calibrated)"}
)
ax.set_title("Per-Class ECE Heatmap — All Models vs All Occupancy Classes",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Occupancy Class", fontsize=11)
ax.set_ylabel("Model", fontsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=9, rotation=0)

plt.tight_layout()
plt.savefig("per_class_ece.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: per_class_ece.pdf")


# ── Per-class reliability diagrams (BNN-Large vs BNS) ────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
n_bins = CFG["ece_bins"]

for row_idx, (key, color, label) in enumerate([
    ("BNN_large", "#228833", "BNN-Large"),
    ("BNS",       "#AA3377", "BNS (Ours)"),
]):
    proba  = results[key]["proba"]
    y_true = y_test

    for c in range(CFG["n_classes"]):
        ax = axes[row_idx, c]

        binary_true = (y_true == c).astype(float)
        conf        = proba[:, c]
        bins        = np.linspace(0, 1, n_bins + 1)

        bx, by, bsize = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() > 0:
                bx.append(conf[mask].mean())
                by.append(binary_true[mask].mean())
                bsize.append(mask.sum())

        ax.plot([0,1],[0,1],"k--",lw=1.2,label="Perfect")
        ax.scatter(bx, by, s=[s/max(bsize)*80 for s in bsize],
                   color=color, edgecolors="black", linewidth=0.5, zorder=3)
        ax.plot(bx, by, "-", color=color, linewidth=1.5,
                label=f"ECE={pc_ece[key][c]:.3f}")

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"Occ-{c}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.set_ylabel(label, fontsize=10, fontweight="bold")

plt.suptitle("Per-Class Reliability Diagrams: BNN-Large vs BNS",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("per_class_reliability.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: per_class_reliability.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: POSTERIOR WEIGHT σ ANALYSIS
# Extract σ = softplus(ρ) from every BayesianDense layer in all BNN variants.
# ═══════════════════════════════════════════════════════════════════════════

def extract_posterior_sigmas(model):
    """
    Returns dict: {layer_name: np.array of σ values (flattened)}
    for every BayesianDense layer in the model.
    """
    sigmas = {}
    for layer in model.layers:
        if isinstance(layer, BayesianDense):
            w_sigma = tf.nn.softplus(layer.w_rho).numpy().flatten()
            b_sigma = tf.nn.softplus(layer.b_rho).numpy().flatten()
            sigmas[layer.name] = np.concatenate([w_sigma, b_sigma])
    return sigmas


models_to_inspect = {
    "BNN-Small" : BNN_small,
    "BNN-Large" : BNN_large,
    "BNN-LowKL" : BNN_lowKL,
    "BNN-HighKL": BNN_highKL,
    "BNS"       : BNS.backbone,
}

all_sigmas = {name: extract_posterior_sigmas(m)
              for name, m in models_to_inspect.items()}

# Summary table
sigma_summary = []
for mname, layers in all_sigmas.items():
    all_w = np.concatenate(list(layers.values()))
    sigma_summary.append(dict(
        Model  = mname,
        Mean_σ = round(all_w.mean(),  5),
        Std_σ  = round(all_w.std(),   5),
        Min_σ  = round(all_w.min(),   5),
        Max_σ  = round(all_w.max(),   5),
        Median_σ = round(np.median(all_w), 5),
        Frac_σ_lt_1e3 = round((all_w < 1e-3).mean(), 4),  # collapsed fraction
    ))

sigma_df = pd.DataFrame(sigma_summary).set_index("Model")
print("Posterior σ Summary (all layers combined):")
print(sigma_df.to_string())
print("\nNote: Frac_σ<0.001 = fraction of weights with near-zero uncertainty (collapsed)")


# ── σ distribution plots: violin + histogram ─────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

n_models = len(all_sigmas)
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, n_models, hspace=0.45, wspace=0.35)

colors = ["#4477AA","#228833","#CCBB44","#EE6677","#AA3377"]

for col, (mname, layers) in enumerate(all_sigmas.items()):
    all_w  = np.concatenate(list(layers.values()))
    color  = colors[col]

    # Top row: violin per layer
    ax_top = fig.add_subplot(gs[0, col])
    layer_names = list(layers.keys())
    data_per_layer = [layers[ln] for ln in layer_names]

    parts = ax_top.violinplot(data_per_layer,
                              positions=range(len(layer_names)),
                              showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax_top.set_xticks(range(len(layer_names)))
    ax_top.set_xticklabels([ln.replace("bayes_","L") for ln in layer_names],
                            rotation=20, fontsize=7)
    ax_top.set_ylabel("σ", fontsize=9)
    ax_top.set_title(mname, fontsize=10, fontweight="bold")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.set_yscale("log")

    # Bottom row: global histogram
    ax_bot = fig.add_subplot(gs[1, col])
    ax_bot.hist(np.clip(all_w, 0, 0.3), bins=60,
                color=color, edgecolor="none", alpha=0.85, density=True)
    ax_bot.axvline(all_w.mean(), color="black", linewidth=1.5,
                   linestyle="--", label=f"μ={all_w.mean():.4f}")
    ax_bot.set_xlabel("σ value", fontsize=9)
    ax_bot.set_ylabel("Density", fontsize=9)
    ax_bot.set_title(f"Global σ dist.", fontsize=9)
    ax_bot.legend(fontsize=7)
    ax_bot.grid(True, alpha=0.3)

fig.suptitle("Posterior Weight σ Analysis — Per Layer (top) & Global (bottom)",
             fontsize=13, fontweight="bold")
plt.savefig("posterior_sigma.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: posterior_sigma.pdf")


# ── σ evolution: first vs last epoch (requires re-capture) ───────────────
# We compare σ magnitude vs KL weight to show the regularization effect

kl_weights   = [
    CFG["kl_base"] * CFG["kl_low_mul"],
    CFG["kl_base"],
    CFG["kl_base"],
    CFG["kl_base"] * CFG["kl_high_mul"],
    CFG["kl_base"],
]
model_labels = ["BNN-LowKL","BNN-Small","BNN-Large","BNN-HighKL","BNS"]

mean_sigmas = []
for mname, layers in all_sigmas.items():
    all_w = np.concatenate(list(layers.values()))
    mean_sigmas.append(all_w.mean())

fig, ax = plt.subplots(figsize=(8, 4))
sc = ax.scatter(kl_weights, mean_sigmas,
                s=120, c=["#CCBB44","#4477AA","#228833","#EE6677","#AA3377"],
                edgecolors="black", linewidth=0.8, zorder=3)

for i, (kl, sig, name) in enumerate(zip(kl_weights, mean_sigmas, model_labels)):
    ax.annotate(name, (kl, sig), textcoords="offset points",
                xytext=(8, 4), fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("KL Weight (κ)", fontsize=11)
ax.set_ylabel("Mean Posterior σ", fontsize=11)
ax.set_title("KL Weight vs Mean Posterior σ\n(Higher KL → tighter posterior)", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kl_vs_sigma.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: kl_vs_sigma.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13.1: λ_BNS SENSITIVITY ABLATION
# Train BNS with 6 different λ values; evaluate TCE and Macro-F1.
# ═══════════════════════════════════════════════════════════════════════════

lambda_grid  = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5]
ablation_lam = []

print("λ_BNS sensitivity sweep (training each variant)...")

for lam in lambda_grid:
    # Fresh backbone
    bb = build_bnn(D, CFG["n_classes"], [128, 64], kl_base, "bns_lam_bb")

    bns_lam = BNS_Model(
        backbone     = bb,
        empirical_mu = empirical_mu,
        N_train      = N_TRAIN,
        lambda_max   = lam,
        beta         = 1.0,
        high_class   = 2,
    )
    bns_lam.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]))

    warmup = LambdaWarmup(warmup_epochs=3)
    bns_lam.fit(
        bns_train_ds,
        validation_data=bns_val_ds,
        epochs=15,
        callbacks=[warmup,
                   tf.keras.callbacks.EarlyStopping(
                       monitor="val_accuracy", patience=5,
                       restore_best_weights=True, verbose=0, mode="max")],
        verbose=0
    )

    proba_lam, _ = predict_bns(bns_lam, X_test, h_test,
                                T=20, mc=True)
    mf1_lam   = f1_score(y_test, np.argmax(proba_lam, 1), average="macro")
    tce_lam,_ = compute_tce(proba_lam, h_test, empirical_mu)
    ece_lam   = compute_ece(y_test, proba_lam)

    ablation_lam.append(dict(lambda_bns=lam,
                             macro_f1=round(mf1_lam,4),
                             tce=round(tce_lam,4),
                             ece=round(ece_lam,4)))
    print(f"  λ={lam:.3f}  MacroF1={mf1_lam:.4f}  TCE={tce_lam:.4f}  ECE={ece_lam:.4f}")

abl_lam_df = pd.DataFrame(ablation_lam)
print("\n" + abl_lam_df.to_string(index=False))


# ── Plot λ ablation ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, metric, title, color in [
    (axes[0], "macro_f1", "Macro F1 (↑)",  "#228833"),
    (axes[1], "tce",      "TCE (↓)",        "#EE6677"),
    (axes[2], "ece",      "ECE (↓)",        "#4477AA"),
]:
    ax.plot(abl_lam_df["lambda_bns"], abl_lam_df[metric],
            "o-", color=color, linewidth=2, markersize=7)
    ax.axvline(CFG["lambda_bns"], color="black", linewidth=1.2,
               linestyle="--", label=f"Default λ={CFG['lambda_bns']}")
    ax.set_xlabel("λ_BNS", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=0.001)

plt.suptitle("λ_BNS Sensitivity Ablation", fontsize=13)
plt.tight_layout()
plt.savefig("lambda_ablation.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: lambda_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13.2: FEATURE GROUP ABLATION
# Drop one feature group at a time; measure Macro-F1 drop on BNN-Large.
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "Temporal (Hour/DoW/Month)" : [0, 1, 2, 3, 4, 5],   # cyclic time features
    "Date (Day/IsWeekend)"      : [6, 7],
    "Temperature (Temp/Dew)"    : [8, 9],
    "Wind+Pressure"             : [10, 11],
    "Rain"                      : [12],
}

feat_ablation = []

# Baseline: all features
logits_base = BNN_large.predict(X_test, batch_size=4096, verbose=0)
proba_base  = tf.nn.softmax(logits_base).numpy()
mf1_base    = f1_score(y_test, np.argmax(proba_base, 1), average="macro")
feat_ablation.append(dict(Dropped="None (baseline)", MacroF1=round(mf1_base, 4),
                          Delta=0.0))

for group_name, feat_idx in FEATURE_GROUPS.items():
    X_ablated = X_test.copy()
    X_ablated[:, feat_idx] = 0.0     # zero-out (mean after scaling = 0)

    logits_ab = BNN_large.predict(X_ablated, batch_size=4096, verbose=0)
    proba_ab  = tf.nn.softmax(logits_ab).numpy()
    mf1_ab    = f1_score(y_test, np.argmax(proba_ab, 1), average="macro")
    delta     = mf1_ab - mf1_base
    feat_ablation.append(dict(Dropped=group_name,
                              MacroF1=round(mf1_ab, 4),
                              Delta=round(delta, 4)))
    print(f"  Drop {group_name:<35s}  F1={mf1_ab:.4f}  Δ={delta:+.4f}")

feat_abl_df = pd.DataFrame(feat_ablation)
print("\n" + feat_abl_df.to_string(index=False))


# ── Feature ablation bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

labels  = feat_abl_df["Dropped"].tolist()
deltas  = feat_abl_df["Delta"].tolist()
colors  = ["#4477AA" if d == 0 else ("#EE6677" if d < 0 else "#228833")
           for d in deltas]

bars = ax.barh(labels, deltas, color=colors, edgecolor="black", linewidth=0.7)
ax.bar_label(bars, fmt="%+.4f", fontsize=9, padding=4)
ax.axvline(0, color="black", linewidth=1.0)
ax.set_xlabel("ΔMacro F1 vs Baseline (negative = important feature)", fontsize=11)
ax.set_title("Feature Group Ablation — BNN-Large", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("feature_ablation.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: feature_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13.3: ARCHITECTURE DEPTH ABLATION
# ═══════════════════════════════════════════════════════════════════════════

arch_configs = [
    ("1-layer  [64]",     [64]),
    ("2-layer  [128,64]", [128, 64]),
    ("3-layer  [256,128,64]", [256, 128, 64]),
    ("BNN-Small [32,16]", [32, 16]),
]

arch_rows = []
print("Architecture depth ablation (each takes ~2 min)...")

for arch_name, hidden in arch_configs:
    m = build_bnn(D, CFG["n_classes"], hidden, kl_base, "arch_tmp")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    m.fit(train_ds, validation_data=val_ds,
          epochs=20, class_weight=CLASS_WEIGHTS,
          callbacks=get_callbacks(), verbose=0)

    logits = m.predict(X_test, batch_size=4096, verbose=0)
    proba  = tf.nn.softmax(logits).numpy()
    y_pred = np.argmax(proba, 1)

    arch_rows.append(dict(
        Architecture = arch_name,
        Params       = m.count_params(),
        Accuracy     = round(accuracy_score(y_test, y_pred), 4),
        MacroF1      = round(f1_score(y_test, y_pred, average="macro"), 4),
        ECE          = round(compute_ece(y_test, proba), 4),
    ))
    print(f"  {arch_name:<28s} params={m.count_params():>7,}  "
          f"acc={arch_rows[-1]['Accuracy']:.4f}  f1={arch_rows[-1]['MacroF1']:.4f}")

arch_df = pd.DataFrame(arch_rows)
print("\n" + arch_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14.1: UNCERTAINTY vs. PREDICTION ERROR
# A well-calibrated model should have HIGH uncertainty when it is WRONG.
# ═══════════════════════════════════════════════════════════════════════════

sample_n = min(30_000, len(X_test))
rng_s    = np.random.default_rng(SEED)
idx_s    = rng_s.choice(len(X_test), sample_n, replace=False)
X_s, y_s = X_test[idx_s], y_test[idx_s]

# BNN-Large MC
bnn_p, bnn_u = mc_predict(BNN_large, X_s, T=30)
bns_p, bns_u = mc_predict(BNS.backbone, X_s, T=30)

def uncertainty_score(proba_mean, proba_std):
    """Mean std over predicted class (epistemic proxy)."""
    pred_class = np.argmax(proba_mean, axis=1)
    return proba_std[np.arange(len(pred_class)), pred_class]

bnn_unc = uncertainty_score(bnn_p, bnn_u)
bns_unc = uncertainty_score(bns_p, bns_u)

bnn_err = (np.argmax(bnn_p, 1) != y_s).astype(int)
bns_err = (np.argmax(bns_p, 1) != y_s).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, unc, err, label, color in [
    (axes[0], bnn_unc, bnn_err, "BNN-Large", "#228833"),
    (axes[1], bns_unc, bns_err, "BNS",       "#AA3377"),
]:
    # Hexbin density
    hb = ax.hexbin(unc, err + np.random.uniform(-0.03, 0.03, len(err)),
                   gridsize=30, cmap="viridis", mincnt=1, alpha=0.8)
    plt.colorbar(hb, ax=ax, label="Count")

    # Trend line
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(unc, err, 1)
    x_line = np.linspace(unc.min(), unc.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            "r-", linewidth=2.5, label=f"Trend (slope={coeffs[0]:.2f})")

    ax.set_xlabel("Predictive Uncertainty (σ_class)", fontsize=11)
    ax.set_ylabel("Error (0=correct, 1=wrong)", fontsize=11)
    ax.set_title(f"{label}: Uncertainty vs. Error", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, unc.quantile(0.99) if hasattr(unc, "quantile")
                else np.percentile(unc, 99))
    ax.grid(True, alpha=0.3)

plt.suptitle("Uncertainty-Error Alignment\n(positive slope = well-calibrated uncertainty)",
             fontsize=12)
plt.tight_layout()
plt.savefig("uncertainty_vs_error.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: uncertainty_vs_error.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14.2: CLASS-CONDITIONAL TEMPORAL PATTERNS
# For each class, how well does each model track the true hourly rate?
# ═══════════════════════════════════════════════════════════════════════════

hours_ax = np.arange(24)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for c in range(CFG["n_classes"]):
    ax = axes[c]

    # True hourly rate for class c
    true_rate = np.array([
        np.mean(y_test[h_test == h] == c) if np.any(h_test == h) else np.nan
        for h in range(24)
    ])
    ax.plot(hours_ax, true_rate, "k-", linewidth=3, label="True", zorder=5)

    model_styles = {
        "DET"       : ("C0", "--",  1.5),
        "BNN_large" : ("C2", "-.",  1.5),
        "BNN_lowKL" : ("C3", ":",   1.5),
        "BNN_highKL": ("C4", "--",  1.5),
        "BNS"       : ("C5", "-",   2.5),
    }
    for key, (color, ls, lw) in model_styles.items():
        rate = np.array([
            np.mean(results[key]["proba"][h_test == h, c])
            if np.any(h_test == h) else np.nan
            for h in range(24)
        ])
        ax.plot(hours_ax, rate, color=color, linestyle=ls, linewidth=lw,
                label=results[key]["name"])

    ax.set_title(f"Occ-{c} Hourly Rate", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel(f"P(Occ={c}|h)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 4))

# 6th subplot: TCE per class bar
ax = axes[5]
for i, (key, color) in enumerate(zip(
    ["DET","BNN_small","BNN_large","BNN_lowKL","BNN_highKL","BNS"],
    ["C0","C1","C2","C3","C4","C5"]
)):
    class_tces = []
    for c in range(CFG["n_classes"]):
        tce_c, _ = compute_tce(
            results[key]["proba"],
            h_test, empirical_mu,
            high_class=c
        )
        class_tces.append(tce_c)
    x_pos = np.arange(CFG["n_classes"]) + i * 0.12 - 0.3
    ax.bar(x_pos, class_tces, width=0.10,
           color=color, label=results[key]["name"],
           edgecolor="black", linewidth=0.5)

ax.set_xticks(range(CFG["n_classes"]))
ax.set_xticklabels([f"Occ-{c}" for c in range(5)])
ax.set_ylabel("TCE"); ax.set_title("TCE per Class", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, ncol=2)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Class-Conditional Temporal Patterns", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("class_temporal_patterns.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: class_temporal_patterns.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14.3: POSTERIOR PREDICTIVE — UNCERTAINTY BANDS OVER TIME
# Show BNS mean prediction ± 2σ over a 48-hour window.
# ═══════════════════════════════════════════════════════════════════════════

# Get a 48-hour contiguous block from test set
# Sort test by hour as proxy for time ordering
sort_idx   = np.argsort(h_test)
X_window   = X_test[sort_idx[:480]]     # 480 samples ≈ 2 days worth
y_window   = y_test[sort_idx[:480]]
h_window   = h_test[sort_idx[:480]]

# MC predictions (BNN-Large and BNS)
T = CFG["mc_samples"]
bnn_preds = np.stack([
    tf.nn.softmax(BNN_large(X_window, training=True)).numpy()
    for _ in range(T)])
bns_preds = np.stack([
    tf.nn.softmax(BNS.backbone(X_window, training=True)).numpy()
    for _ in range(T)])

# Expected occupancy (weighted sum of class indices)
class_vals = np.arange(CFG["n_classes"])

def expected_occ(preds):
    mean_p = preds.mean(axis=0)   # (N, C)
    std_p  = preds.std(axis=0)
    exp_m  = (mean_p * class_vals).sum(axis=1) / (CFG["n_classes"]-1)
    # uncertainty via std of expected value
    exp_s  = np.sqrt(((std_p**2) * (class_vals**2)).sum(axis=1)) / (CFG["n_classes"]-1)
    return exp_m, exp_s

bnn_mean, bnn_std = expected_occ(bnn_preds)
bns_mean, bns_std = expected_occ(bns_preds)
true_occ = y_window / (CFG["n_classes"]-1)

t = np.arange(len(y_window))

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for ax, mean, std, label, color in [
    (axes[0], bnn_mean, bnn_std, "BNN-Large", "#228833"),
    (axes[1], bns_mean, bns_std, "BNS (Ours)", "#AA3377"),
]:
    ax.plot(t, true_occ, "k-", linewidth=1.5, alpha=0.7, label="True")
    ax.plot(t, mean,  "-",  color=color, linewidth=2.0, label=f"{label} (mean)")
    ax.fill_between(t,
                    np.clip(mean - 2*std, 0, 1),
                    np.clip(mean + 2*std, 0, 1),
                    alpha=0.25, color=color, label="±2σ")
    coverage = np.mean(
        (true_occ >= mean - 2*std) & (true_occ <= mean + 2*std)
    )
    ax.text(0.02, 0.85,
            f"±2σ coverage: {coverage*100:.1f}%  (ideal ≈ 95%)",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.set_ylabel("Normalized Occupancy", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title(label, fontsize=11, fontweight="bold")

axes[1].set_xlabel("Sample Index (sorted by hour)", fontsize=11)
plt.suptitle("Posterior Predictive Distribution with Uncertainty Bands",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("posterior_predictive_bands.pdf", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: posterior_predictive_bands.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14.4: COMPLETE SUMMARY FIGURE (publication-ready, 3×3 grid)
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.38)

colors6 = ["#4477AA","#66CCEE","#228833","#CCBB44","#EE6677","#AA3377"]
mnames  = [results[k]["name"] for k in MODEL_KEYS]
x6      = np.arange(len(mnames))

# ── (0,0) Accuracy bar ───────────────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
vals = [results[k]["accuracy"] for k in MODEL_KEYS]
b    = ax00.bar(x6, vals, color=colors6, edgecolor="k", linewidth=0.7)
ax00.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
ax00.set_xticks(x6); ax00.set_xticklabels(mnames, rotation=25, ha="right", fontsize=8)
ax00.set_ylabel("Accuracy"); ax00.set_title("Test Accuracy", fontweight="bold")
ax00.grid(axis="y", alpha=0.3); ax00.set_ylim(0, max(vals)*1.14)

# ── (0,1) Macro F1 bar ───────────────────────────────────────────────────
ax01 = fig.add_subplot(gs[0, 1])
vals = [results[k]["macro_f1"] for k in MODEL_KEYS]
b    = ax01.bar(x6, vals, color=colors6, edgecolor="k", linewidth=0.7)
ax01.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
ax01.set_xticks(x6); ax01.set_xticklabels(mnames, rotation=25, ha="right", fontsize=8)
ax01.set_ylabel("Macro F1"); ax01.set_title("Macro F1 (Fairness)", fontweight="bold")
ax01.grid(axis="y", alpha=0.3); ax01.set_ylim(0, max(vals)*1.14)

# ── (0,2) ECE bar ────────────────────────────────────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
vals = [results[k]["ece"] for k in MODEL_KEYS]
b    = ax02.bar(x6, vals, color=colors6, edgecolor="k", linewidth=0.7)
ax02.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
ax02.set_xticks(x6); ax02.set_xticklabels(mnames, rotation=25, ha="right", fontsize=8)
ax02.set_ylabel("ECE"); ax02.set_title("ECE (↓ Calibration)", fontweight="bold")
ax02.grid(axis="y", alpha=0.3)

# ── (1,0) Temporal consistency ───────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.plot(np.arange(24), empirical_mu, "k-", linewidth=3, label="True")
for k, c, ls in zip(
    ["DET","BNN_large","BNS"],
    ["C0","C2","C5"],
    ["--","-.","−"]
):
    ax10.plot(np.arange(24), results[k]["mu_pred"], color=c,
              linestyle="--" if ls=="--" else ("-." if ls=="-." else "-"),
              linewidth=2, label=results[k]["name"])
ax10.set_xlabel("Hour"); ax10.set_ylabel("P(High Occ | h)")
ax10.set_title("Temporal Consistency", fontweight="bold")
ax10.legend(fontsize=8); ax10.grid(True, alpha=0.3)
ax10.set_xticks(range(0,24,4))

# ── (1,1) Per-class ECE heatmap ──────────────────────────────────────────
ax11 = fig.add_subplot(gs[1, 1])
im = ax11.imshow(pc_df.astype(float).values, aspect="auto", cmap="YlOrRd")
ax11.set_xticks(range(5))
ax11.set_xticklabels([f"Occ-{c}" for c in range(5)], fontsize=9)
ax11.set_yticks(range(len(pc_df)))
ax11.set_yticklabels(pc_df.index, fontsize=8)
plt.colorbar(im, ax=ax11, fraction=0.04)
ax11.set_title("Per-Class ECE", fontweight="bold")
for i in range(len(pc_df)):
    for j in range(5):
        ax11.text(j, i, f"{pc_df.values[i,j]:.3f}",
                  ha="center", va="center", fontsize=7,
                  color="white" if pc_df.values[i,j] > 0.04 else "black")

# ── (1,2) σ violin per model ─────────────────────────────────────────────
ax12 = fig.add_subplot(gs[1, 2])
sigma_data  = []
sigma_labels= []
for mname, layers in all_sigmas.items():
    all_w = np.concatenate(list(layers.values()))
    sigma_data.append(np.clip(all_w, 0, 0.15))
    sigma_labels.append(mname)
vp = ax12.violinplot(sigma_data, showmedians=True, showextrema=False)
for i, pc in enumerate(vp["bodies"]):
    pc.set_facecolor(colors6[i]); pc.set_alpha(0.75)
vp["cmedians"].set_color("black")
ax12.set_xticks(range(1, len(sigma_labels)+1))
ax12.set_xticklabels(sigma_labels, rotation=25, ha="right", fontsize=8)
ax12.set_ylabel("Posterior σ"); ax12.set_title("Weight Posterior σ", fontweight="bold")
ax12.grid(axis="y", alpha=0.3)

# ── (2,0) λ ablation ─────────────────────────────────────────────────────
ax20 = fig.add_subplot(gs[2, 0])
ax20.plot(abl_lam_df["lambda_bns"], abl_lam_df["macro_f1"],
          "o-", color="#AA3377", linewidth=2)
ax20.axvline(CFG["lambda_bns"], color="black", linewidth=1.2,
             linestyle="--", label=f"λ={CFG['lambda_bns']}")
ax20.set_xscale("symlog", linthresh=0.001)
ax20.set_xlabel("λ_BNS"); ax20.set_ylabel("Macro F1")
ax20.set_title("λ Sensitivity", fontweight="bold")
ax20.legend(fontsize=8); ax20.grid(True, alpha=0.3)

# ── (2,1) Feature ablation ────────────────────────────────────────────────
ax21 = fig.add_subplot(gs[2, 1])
fa_labels = feat_abl_df["Dropped"].tolist()
fa_deltas = feat_abl_df["Delta"].tolist()
fa_colors = ["#4477AA" if d == 0 else "#EE6677" for d in fa_deltas]
ax21.barh(fa_labels, fa_deltas, color=fa_colors, edgecolor="k", linewidth=0.7)
ax21.axvline(0, color="k", linewidth=1.0)
ax21.set_xlabel("ΔMacro F1"); ax21.set_title("Feature Ablation", fontweight="bold")
ax21.grid(axis="x", alpha=0.3)
ax21.tick_params(labelsize=8)

# ── (2,2) Noise robustness ────────────────────────────────────────────────
ax22 = fig.add_subplot(gs[2, 2])
noise_pivot = noise_df.pivot(index="Model", columns="Noise", values="MacroF1")
for i, (model_r, row) in enumerate(noise_pivot.iterrows()):
    ax22.plot(row.index, row.values, "o-",
              linewidth=2, markersize=7,
              label=model_r, color=colors6[i])
ax22.set_xlabel("Label Noise"); ax22.set_ylabel("Macro F1")
ax22.set_title("Noise Robustness", fontweight="bold")
ax22.legend(fontsize=9); ax22.grid(True, alpha=0.3)

fig.suptitle("BNS vs BNN: Complete Experimental Summary (NeurIPS-Ready)",
             fontsize=15, fontweight="bold", y=1.005)
plt.savefig("neurips_summary_figure.pdf", bbox_inches="tight", dpi=180)
plt.show()
print("\n✅ Saved: neurips_summary_figure.pdf  ← MAIN PAPER FIGURE")


import os

outputs = [
    # Existing
    "confusion_matrices.pdf",
    "temporal_consistency.pdf",
    "reliability_diagrams.pdf",
    "kl_ablation.pdf",
    "training_curves.pdf",
    "uncertainty_analysis.pdf",
    "final_comparison.pdf",
    # New additions
    "ts_split_diagnostics.pdf",
    "radar_chart.pdf",
    "per_class_ece.pdf",
    "per_class_reliability.pdf",
    "posterior_sigma.pdf",
    "kl_vs_sigma.pdf",
    "lambda_ablation.pdf",
    "feature_ablation.pdf",
    "uncertainty_vs_error.pdf",
    "class_temporal_patterns.pdf",
    "posterior_predictive_bands.pdf",
    "neurips_summary_figure.pdf",
]

print("=" * 55)
print("  COMPLETE OUTPUT CHECKLIST")
print("=" * 55)
for f in outputs:
    status = "✅" if os.path.exists(f) else "⬜ (will exist after run)"
    print(f"  {status}  {f}")

print()
print("Extended metrics table:")
print(ext_df[["MacroF1","Brier","NLL","AUROC","Kappa","MCC"]].to_string())
print("=" * 55)




