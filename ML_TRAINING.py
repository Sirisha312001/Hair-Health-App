# ============================================================
# ML_TRAINING.py
# Train hair loss cause model for SAHH Assistant
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer  # for handling NaNs

# ------------------------------------------------------------
# 1. CONFIG – PATHS
# ------------------------------------------------------------

BASE_DIR = r"C:\local disk E\realme files\siri project\Hair project"

# Your cleaned survey file (without a 'cause' column yet)
DATA_PATH = os.path.join(BASE_DIR, "clean_survey_data_final.csv")

# Output ML-ready CSV + model path
ML_READY_CSV = os.path.join(BASE_DIR, "clean_survey_with_cause.csv")
MODEL_PATH = os.path.join(BASE_DIR, "hairloss_cause_model.pkl")

TARGET_COL = "cause"          # label we create
ID_COLS_TO_DROP = ["Timestamp"]

# ------------------------------------------------------------
# 2. LOAD + CLEAN COLUMN NAMES
# ------------------------------------------------------------

print(">>> Loading data...")
df = pd.read_csv(DATA_PATH)

print("Original columns:")
print(df.columns.tolist())

# Strip extra spaces from column names
df.columns = [c.strip() for c in df.columns]
print("\nCleaned columns:")
print(df.columns.tolist())

# Drop obvious ID/timestamp columns if present
drop_cols = [c for c in ID_COLS_TO_DROP if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print("\nDropped columns:", drop_cols)

print("\nData shape after cleaning:", df.shape)

# ------------------------------------------------------------
# 3. CREATE RULE-BASED CAUSE LABEL (TARGET)
# ------------------------------------------------------------

def safe_lower(x):
    return str(x).strip().lower() if pd.notna(x) else ""

# Numeric-like columns → numeric
for col in [
    "Stress Level (0-10)",
    "Sleep Hours",
    "Daily Water Intake (Liters)",
    "Hair Loss Severity (0–10)",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def infer_cause(row):
    """
    Heuristic rules to label each row with a main hair-loss 'cause'.
    These labels are what the model will learn to predict, and they
    must match the names used in the recommendation files.
    """
    scale_type = safe_lower(row.get("Scale Type", ""))
    anti_dandruff = safe_lower(row.get("Anti-Dandruff Shampoo Use", ""))
    water_type = safe_lower(row.get("Water Type at Home", ""))
    water_changed = safe_lower(row.get("Water Type Changed After Coming to USA", ""))
    hair_loss_increased = safe_lower(row.get("Hair Loss Increased After Coming to USA", ""))
    vit_d = safe_lower(row.get("Vitamin D Deficiency", ""))
    stress = row.get("Stress Level (0-10)", np.nan)
    sleep = row.get("Sleep Hours", np.nan)

    # ---- 1) Dandruff / fungal pattern ----
    if (
        "dandruff" in scale_type
        or "flake" in scale_type
        or "itch" in scale_type
        or "fungal" in scale_type
        or "yes" in anti_dandruff
    ):
        return "Dandruff_Fungal"

    # ---- 2) Hard-water damage ----
    if (
        "hard" in water_type
        or ("yes" in water_changed and "yes" in hair_loss_increased)
    ):
        return "Hard_Water_Damage"

    # ---- 3) Vitamin D deficiency pattern ----
    if "yes" in vit_d:
        return "Vitamin_D_Deficiency"

    # ---- 4) Stress-related shedding ----
    if (pd.notna(stress) and stress >= 7) or (pd.notna(sleep) and sleep <= 5):
        return "Stress_Shedding"

    # ---- 5) Fallback / mixed ----
    return "Mixed_Other"


print("\n>>> Creating 'cause' labels using heuristic rules...")
df[TARGET_COL] = df.apply(infer_cause, axis=1)

print("\nCause distribution (new target column):")
print(df[TARGET_COL].value_counts())

# Save ML-ready dataset with cause labels (for debugging / EDA)
df.to_csv(ML_READY_CSV, index=False)
print(f"\nSaved ML-ready dataset with '{TARGET_COL}' to:\n{ML_READY_CSV}")

# ------------------------------------------------------------
# 4. SPLIT FEATURES / TARGET
# ------------------------------------------------------------

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ------------------------------------------------------------
# 5. PREPROCESSOR + MODELS (WITH IMPUTERS & BALANCED CLASSES)
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 👇 main change: class_weight="balanced" so it doesn’t always pick the majority class
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
    ),
}

# ------------------------------------------------------------
# 6. TRAIN + EVALUATE
# ------------------------------------------------------------

results = []

for name, clf in models.items():
    print("\n" + "=" * 60)
    print(f">>> Training: {name}")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:     {acc:.4f}")
    print(f"F1-weighted:  {f1w:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    results.append(
        {"name": name, "pipeline": pipe, "accuracy": acc, "f1_weighted": f1w}
    )

# Pick best by F1-weighted
results_sorted = sorted(results, key=lambda d: d["f1_weighted"], reverse=True)
best = results_sorted[0]
best_name = best["name"]
best_model = best["pipeline"]

print("\n" + "=" * 60)
print("MODEL COMPARISON:")
for r in results_sorted:
    print(
        f"{r['name']:<20}  Acc: {r['accuracy']:.4f}  "
        f"F1-weighted: {r['f1_weighted']:.4f}"
    )

print(f"\n>>> Selected best model: {best_name}")

# ------------------------------------------------------------
# 7. REFIT BEST MODEL ON FULL DATA + SAVE
# ------------------------------------------------------------

print("\n>>> Refitting best model on FULL data...")
best_model.fit(X, y)

os.makedirs(BASE_DIR, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"\n>>> Saved best model to:\n{MODEL_PATH}")

# For your debugging / report
classes_ = best_model.named_steps["clf"].classes_
print("\nModel classes (hair loss causes):")
print(classes_)
print("\nTraining complete.")
