# ==========================================================
# Arohi Hair Health Assistant (Streamlit App)
# Final integrated version
# ==========================================================

import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np        

# ----------------------------------------------------------
# 0. PATHS – RELATIVE TO THIS FILE (NO C:\ PATHS)
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "hairloss_cause_model.pkl")
PRODUCTS_PATH = os.path.join(BASE_DIR, "products_master_clean_small.csv")
FOOD_RECOMMENDATIONS_PATH = os.path.join(
    BASE_DIR, "Hairfall_Food_Recommendations.xlsx"
)
SUPPLEMENT_GUIDE_PATH = os.path.join(
    BASE_DIR, "Copy of Hairfall_Supplement_Guide.xlsx"
)

#  Arohi icon (relative path, like SAHH example)
IMAGE_AROHI_PATH = os.path.join(
    BASE_DIR, "AROHI_ICON.png"
)

# These are the feature columns used when training hairloss_cause_model.pkl
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Country of Origin",
    "State in USA",
    "Years in USA",
    "Water Type at Home",
    "Water Type Changed After Coming to USA",
    "Diet Change After Moving",
    "Daily Water Intake (Liters)",
    "Stress Level (0-10)",
    "Sleep Hours",
    "Hair Type",
    "Scale Type",
    "Wash Frequency Per Week",
    "Do You Oil Your Hair?",
    "Heat Tools Usage",
    "Hair Color / Bleach",
    "Hijab or Turban Use",
    "Use of Hair Styling Products",
    "Iron Deficiency",
    "Vitamin D Deficiency",
    "Thyroid Issues",
    "Recent Major Illness",
    "Hair Loss Severity (0–10)",
    "Hair Loss Increased After Coming to USA",
    "Hair Loss Pattern",
    "When Did Hair Loss Start?",
    "Anti-Dandruff Shampoo Use",
    "Top 3 Hair Products You Use",
    "Budget for Hair Products",
    "Consent to Use Data for Academic Research",
]

# ----------------------------------------------------------
# 1. PAGE CONFIG – MUST BE FIRST STREAMLIT CALL
# ----------------------------------------------------------
st.set_page_config(
    page_title="Arohi Hair Health Assistant",
    page_icon=IMAGE_AROHI_PATH if os.path.exists(IMAGE_AROHI_PATH) else "💇‍♀️",
    layout="wide",
)

# ----------------------------------------------------------
# 2. GLOBAL CSS – COLORS, FONTS, BUTTONS
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    /* APP BACKGROUND */
    .stApp {
        background-color: #f3fbff;
        color: #082032;
        font-family: "Segoe UI", system-ui, sans-serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        background-color: transparent;
    }

    section[data-testid="stSidebar"] {
        background-color: #004d40;
    }
    section[data-testid="stSidebar"] * {
        color: #f3f6f7 !important;
        font-weight: 600;
    }

    .sahh-section-label {
        color: #4e342e !important;
        font-weight: 800 !important;
        margin-bottom: 0.25rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #004d40 !important;
        font-weight: 800 !important;
    }

    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stCheckbox"] label p {
        color: #003c3c !important;
        font-weight: 700 !important;
    }

    .sahh-card, .sahh-note, .sahh-warning, .sahh-info {
        color: #1b1b1b !important;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #c0c7cf;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .sahh-note-yellow {
        background-color: #ffe39b !important;
        color: #3e2723 !important;
    }
    .sahh-card-pink {
        background-color: #ffc6c6 !important;
        color: #3e2723 !important;
    }
    .sahh-card-blue {
        background-color: #cfe4ff !important;
        color: #003049 !important;
    }
    .sahh-card-green {
        background-color: #c6f6d5 !important;
        color: #064420 !important;
    }

    .sahh-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background-color: #004d40;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.9rem;
    }

    .sahh-steps {
        font-size: 0.85rem;
        line-height: 1.4;
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        background-color: #ffe6c4;
        color: #4e342e !important;
        font-weight: 700;
    }

    .stButton > button {
        background-color: #e53935 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        border: 1px solid #b71c1c !important;
        font-weight: 700 !important;
        padding: 0.4rem 1.3rem !important;
    }
    .stButton > button:hover {
        background-color: #ff5252 !important;
        color: #ffffff !important;
        border-color: #b71c1c !important;
    }

    .sahh-cause-tag {
        display:inline-block;
        padding:0.4rem 0.9rem;
        border-radius:999px;
        background-color:#004d40;
        color:#ffffff;
        font-weight:800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# 3. CACHED LOADERS
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Please make sure 'hairloss_cause_model.pkl' "
            "is in the same folder as hairapp.py."
        )
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load hair loss model. Error: {e}")
        return None


@st.cache_data
def load_products():
    if not os.path.exists(PRODUCTS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(PRODUCTS_PATH)
    for col in ["product_name", "brand", "category", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data
def load_food_recs():
    if not os.path.exists(FOOD_RECOMMENDATIONS_PATH):
        return pd.DataFrame()
    df = pd.read_excel(FOOD_RECOMMENDATIONS_PATH, sheet_name=0)
    df.columns = df.columns.str.lower().str.strip()
    return df


@st.cache_data
def load_supp_recs():
    if not os.path.exists(SUPPLEMENT_GUIDE_PATH):
        return pd.DataFrame()
    df = pd.read_excel(SUPPLEMENT_GUIDE_PATH, sheet_name=0)
    df.columns = df.columns.str.lower().str.strip()
    return df


# ----------------------------------------------------------
# 4. MAPPINGS & HELPERS
# ----------------------------------------------------------
cause_to_needs = {
    "Iron_Deficiency": ["strengthening", "general_hair_health", "hair_growth"],
    "Vitamin_D_Deficiency": ["scalp_nourish", "general_hair_health"],
    "Zinc_Deficiency": ["strengthening", "anti_hair_fall"],
    "Biotin_Deficiency": ["hair_growth", "strengthening"],
    "Vitamin_B12_Deficiency": ["strengthening", "hair_growth"],
    "Protein_Deficiency": ["strengthening", "thickening"],
    "Omega3_Deficiency": ["scalp_nourish", "anti_inflammation"],
    "Dandruff_Fungal": ["anti_dandruff", "scalp_care"],
    "Dry_Scalp": ["hydrating", "scalp_nourish"],
    "Oily_Scalp": ["clarifying", "oil_control"],
    "Seborrheic_Dermatitis": ["anti_dandruff", "scalp_treatment"],
    "Stress_Sleep": ["anti_hair_fall", "soothing"],
    "Stress_Shedding": ["anti_hair_fall", "soothing"],
    "Low_Water_Intake": ["hydrating", "general_hair_health"],
    "Crash_Dieting": ["strengthening", "hair_growth"],
    "Junk_Food_High_Sugar": ["detox", "general_hair_health"],
    "Heat_Styling": ["damage_repair", "strengthening"],
    "Chemical_Treatments": ["damage_repair", "scalp_protection"],
    "Tight_Hairstyles": ["anti_breakage", "scalp_relief"],
    "Coloring_Bleaching": ["damage_repair", "hydrating"],
    "Hard_Water_Damage": ["clarifying", "damage_repair"],
    "Pollution": ["detox", "scalp_protection"],
    "Seasonal_Shedding": ["general_hair_health", "anti_hair_fall"],
    "Thyroid_Disorders": ["anti_hair_fall", "strengthening"],
    "PCOS_Hormonal": ["anti_hair_fall", "scalp_nourish"],
    "Postpartum": ["hair_growth", "general_hair_health"],
    "Anemia": ["strengthening", "hair_growth"],
    "Genetic_Androgenetic": ["thickening", "volumizing"],
    "Lifestyle_Diet_Change": ["general_hair_health"],
    "Mixed_Other": ["general_hair_health"],
    "Unknown": ["general_hair_health"],
}

need_to_keywords = {
    "anti_dandruff": ["dandruff", "anti-dandruff", "ketoconazole", "scalp"],
    "scalp_care": ["scalp care", "scalp scrub", "scalp serum"],
    "scalp_nourish": ["scalp oil", "scalp nourish", "soothe scalp"],
    "clarifying": ["clarifying", "build-up", "detox shampoo"],
    "oil_control": ["oil control", "oily scalp", "sebum"],
    "damage_repair": ["repair", "bond", "damaged hair", "reconstruct"],
    "anti_breakage": ["breakage", "split ends", "strengthen"],
    "detox": ["detox", "anti-pollution"],
    "general_hair_health": ["shampoo", "conditioner", "hair oil", "hair mask"],
    "hair_growth": ["growth", "regrowth", "biotin", "stimulating"],
    "anti_inflammation": ["soothing", "calm scalp"],
    "strengthening": ["protein", "strength", "fortifying"],
    "anti_hair_fall": ["hair fall", "fall control", "root strengthen"],
    "soothing": ["soothing", "calm"],
    "thickening": ["thickening", "density"],
    "volumizing": ["volume", "volumizing"],
    "scalp_protection": ["protect", "shield"],
    "hydrating": ["hydrating", "moisture", "nourishing"],
}

CAUSE_EXPLANATIONS = {
    "Dandruff_Fungal": "Your answers suggest flakes/itching or benefit from anti-dandruff / anti-fungal care.",
    "Hard_Water_Damage": "Your pattern fits hard-water related damage — often worse after moving and using different water.",
    "Vitamin_D_Deficiency": "Low sunlight or Vitamin D history can weaken roots and slow regrowth.",
    "Stress_Shedding": "High stress and/or low sleep can trigger stress-related shedding.",
    "Mixed_Other": "Your pattern looks mixed – more than one factor may be contributing to hair loss.",
}

def build_model_input(user_answers: dict) -> pd.DataFrame:
    """
    Build a one-row DataFrame with exactly the training feature columns.
    Missing values (None) will be handled by the model's imputers.
    """
    row = {col: user_answers.get(col, None) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def _build_cause_pattern(cause: str) -> str:
    """Regex to match cause names like 'Dandruff_Fungal' in Excel files."""
    base = cause.replace("_", " ").strip()
    parts = base.split()
    if len(parts) >= 2:
        return rf"{parts[0]}[ _/]+{parts[1]}"
    return base


def get_foods_for_cause(cause: str, df: pd.DataFrame, top_n: int = 5):
    if df.empty or "cause" not in df.columns:
        return []
    pattern = _build_cause_pattern(cause)
    mask = df["cause"].astype(str).str.contains(
        pattern, case=False, regex=True, na=False
    )
    rows = df.loc[mask].head(top_n)
    out = []
    for _, r in rows.iterrows():
        nutrient = str(r.get("nutrient", "")).strip()
        foods = str(r.get("foods", "")).strip()
        if not foods:
            continue
        if nutrient:
            out.append(f"{foods} – focus on **{nutrient}**.")
        else:
            out.append(foods)
    return out


def get_supps_for_cause(cause: str, df: pd.DataFrame, top_n: int = 5):
    if df.empty or "cause" not in df.columns:
        return []
    pattern = _build_cause_pattern(cause)
    mask = df["cause"].astype(str).str.contains(
        pattern, case=False, regex=True, na=False
    )
    rows = df.loc[mask].head(top_n)
    out = []
    for _, r in rows.iterrows():
        supp = str(r.get("supplement", "")).strip()
        suggestion = str(r.get("suggestion", "")).strip()
        caution = str(r.get("caution", "")).strip()
        parts = []
        if supp:
            parts.append(f"**{supp}**")
        if suggestion:
            parts.append(suggestion)
        if caution:
            parts.append(f"⚠ {caution}")
        if parts:
            out.append(" – ".join(parts))
    return out


def get_products_for_cause(
    cause: str, products_df: pd.DataFrame, budget_band: str, source: str, top_n: int = 5
):
    if products_df.empty:
        return pd.DataFrame()

    df = products_df.copy()
    df["category"] = df["category"].astype(str)
    df["product_name"] = df["product_name"].astype(str)

    # 🔹 Strong filter: ONLY HAIR-RELATED PRODUCTS
    hair_mask = (
        df["category"].str.contains(
            r"hair|scalp|shampoo|conditioner|mask|oil|leave\-in|serum|tonic|treatment",
            case=False,
            na=False,
        )
        | df["product_name"].str.contains(
            r"hair|scalp|shampoo|conditioner|mask|oil|leave\-in|tonic|treatment",
            case=False,
            na=False,
        )
    )

    not_hair_mask = df["category"].str.contains(
        r"face|skin|moisturizer|cleanser|serum|makeup|toner|body|lotion|sunscreen",
        case=False,
        na=False,
    )

    df = df[hair_mask & ~not_hair_mask]

    needs = cause_to_needs.get(cause, ["general_hair_health"])
    mask = pd.Series(False, index=df.index)
    for need in needs:
        for kw in need_to_keywords.get(need, []):
            m = df["category"].str.contains(kw, case=False, na=False) | df[
                "product_name"
            ].str.contains(kw, case=False, na=False)
            mask |= m

    df = df[mask]

    if "budget_band" in df.columns and budget_band:
        df = df[df["budget_band"] == budget_band]

    if source and "source" in df.columns:
        df = df[df["source"] == source]

    if "rating" in df.columns and "price" in df.columns:
        df = df.sort_values(["rating", "price"], ascending=[False, True])

    keep_cols = [
        c
        for c in [
            "product_name",
            "brand",
            "category",
            "price",
            "rating",
            "budget_band",
            "source",
            "url",
        ]
        if c in df.columns
    ]
    return df[keep_cols].head(top_n)


# -------- Rule-based fallback cause logic (same as training script) ----------
def _safe_lower_app(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def infer_cause_from_answers(answers: dict) -> str:
    """
    Rule-based fallback cause.
    Uses the same logic as ML_TRAINING.py so labels match:
    Dandruff_Fungal, Hard_Water_Damage, Vitamin_D_Deficiency,
    Stress_Shedding, Mixed_Other.
    """
    scale_type = _safe_lower_app(answers.get("Scale Type"))
    anti_dandruff = _safe_lower_app(answers.get("Anti-Dandruff Shampoo Use"))
    water_type = _safe_lower_app(answers.get("Water Type at Home"))
    water_changed = _safe_lower_app(
        answers.get("Water Type Changed After Coming to USA")
    )
    hair_loss_increased = _safe_lower_app(
        answers.get("Hair Loss Increased After Coming to USA")
    )
    vit_d = _safe_lower_app(answers.get("Vitamin D Deficiency"))
    stress = answers.get("Stress Level (0-10)")
    sleep = answers.get("Sleep Hours")

    # make sure stress / sleep are numeric for comparisons
    try:
        stress_val = float(stress) if stress is not None else np.nan
    except Exception:
        stress_val = np.nan
    try:
        sleep_val = float(sleep) if sleep is not None else np.nan
    except Exception:
        sleep_val = np.nan

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
    if (pd.notna(stress_val) and stress_val >= 7) or (
        pd.notna(sleep_val) and sleep_val <= 5
    ):
        return "Stress_Shedding"

    # ---- 5) Fallback / mixed ----
    return "Mixed_Other"


# -------- Meal scoring helpers (handles long labels) ----------
def _normalize_meal_group_label(label: str) -> str:
    if label.startswith("**Protein**"):
        return "Protein"
    if label.startswith("**Veg & Fruits**"):
        return "Veg & Fruits"
    if label.startswith("**Whole Grains**"):
        return "Whole Grains"
    if label.startswith("**Healthy Fats & Nuts**"):
        return "Healthy Fats & Nuts"
    if label.startswith("**Fluids**"):
        return "Fluids"
    if label.startswith("**Fermented Foods**"):
        return "Fermented Foods"
    if label.startswith("**Sugary / Processed**"):
        return "Sugary / Processed"
    return label


def score_single_meal(groups):
    """Simple rule-based meal score."""
    if groups is None:
        groups = []

    base_groups = [_normalize_meal_group_label(g) for g in groups]

    score = 0.0
    pos = {
        "Protein": 2.0,
        "Veg & Fruits": 2.0,
        "Whole Grains": 1.0,
        "Healthy Fats & Nuts": 1.0,
        "Fluids": 0.5,
        "Fermented Foods": 1.0,
    }
    neg = {
        "Sugary / Processed": -2.0,
    }

    for g, v in pos.items():
        if g in base_groups:
            score += v
    for g, v in neg.items():
        if g in base_groups:
            score += v

    return score


def evaluate_daily_meals(meal_groups_list):
    scores = [score_single_meal(gs) for gs in meal_groups_list]
    total = sum(scores)
    if total >= 8:
        verdict = "Good base for hair health."
    elif total >= 4:
        verdict = "Okay, can add more protein & veggies."
    else:
        verdict = "Needs work – focus on protein, iron & veggies."
    return total, verdict


def evaluate_chronic_flags(flags_dict):
    """Return a compact summary string for chronic check."""
    # PCOS-like
    pcos_score = 0
    if flags_dict["irregular_periods"]:
        pcos_score += 2
    if flags_dict["heavy_light"]:
        pcos_score += 1
    if flags_dict["acne_hair"]:
        pcos_score += 1
    if flags_dict["pcos_weight"]:
        pcos_score += 1
    if flags_dict["fam_pcos"]:
        pcos_score += 1

    # Thyroid-like
    thy_score = 0
    if flags_dict["unexp_weight"]:
        thy_score += 2
    if flags_dict["temp_intol"]:
        thy_score += 1
    if flags_dict["fatigue"]:
        thy_score += 1
    if flags_dict["palp_anx"]:
        thy_score += 1
    if flags_dict["fam_thy"]:
        thy_score += 1

    # Autoimmune-like
    auto_score = 0
    if flags_dict["joint_pain"]:
        auto_score += 2
    if flags_dict["sun_rash"]:
        auto_score += 2
    if flags_dict["mouth_ulcers"]:
        auto_score += 1
    if flags_dict["fever"]:
        auto_score += 1
    if flags_dict["fam_auto"]:
        auto_score += 1

    def label(s):
        if s >= 5:
            return "HIGH"
        elif s >= 3:
            return "MODERATE"
        elif s >= 1:
            return "LOW"
        return "NONE"

    p_label = label(pcos_score)
    t_label = label(thy_score)
    a_label = label(auto_score)

    parts = []
    if p_label != "NONE":
        parts.append(f"PCOS-like pattern: **{p_label}** (score {pcos_score}).")
    if t_label != "NONE":
        parts.append(f"Thyroid-like pattern: **{t_label}** (score {thy_score}).")
    if a_label != "NONE":
        parts.append(f"Autoimmune-like pattern: **{a_label}** (score {auto_score}).")
    if not parts:
        parts.append(
            "No strong PCOS, thyroid or autoimmune pattern seen here. If worried, please see a doctor."
        )

    return "\n".join(parts)

# ----------------------------------------------------------
# 4b. RULE-BASED CAUSE ENGINE (APP SIDE)
# ----------------------------------------------------------

def _safe_lower_app(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def infer_cause_from_answers(answers: dict) -> str:
    """
    Rule-based cause engine using quiz answers.
    Returns one of the labels used in cause_to_needs.
    """

    # ----- read & normalize fields -----
    scale_type = _safe_lower_app(answers.get("Scale Type"))
    anti_dandruff = _safe_lower_app(answers.get("Anti-Dandruff Shampoo Use"))
    water_type = _safe_lower_app(answers.get("Water Type at Home"))
    water_changed = _safe_lower_app(
        answers.get("Water Type Changed After Coming to USA")
    )
    hair_loss_increased = _safe_lower_app(
        answers.get("Hair Loss Increased After Coming to USA")
    )
    vit_d = _safe_lower_app(answers.get("Vitamin D Deficiency"))
    iron_def = _safe_lower_app(answers.get("Iron Deficiency"))
    thyroid = _safe_lower_app(answers.get("Thyroid Issues"))
    recent_illness = _safe_lower_app(answers.get("Recent Major Illness"))
    diet_change = _safe_lower_app(answers.get("Diet Change After Moving"))
    hair_color = _safe_lower_app(answers.get("Hair Color / Bleach"))
    heat_tools = _safe_lower_app(answers.get("Heat Tools Usage"))
    wash_freq = _safe_lower_app(answers.get("Wash Frequency Per Week"))
    water_intake = answers.get("Daily Water Intake (Liters)")
    stress = answers.get("Stress Level (0-10)")
    sleep = answers.get("Sleep Hours")
    pattern = _safe_lower_app(answers.get("Hair Loss Pattern"))
    gender = _safe_lower_app(answers.get("Gender"))
    hijab = _safe_lower_app(answers.get("Hijab or Turban Use"))
    styling_prod = _safe_lower_app(answers.get("Use of Hair Styling Products"))

    # numeric versions
    try:
        stress_val = float(stress) if stress is not None else np.nan
    except Exception:
        stress_val = np.nan

    try:
        sleep_val = float(sleep) if sleep is not None else np.nan
    except Exception:
        sleep_val = np.nan

    try:
        water_val = float(water_intake) if water_intake is not None else np.nan
    except Exception:
        water_val = np.nan

    # ----- score each possible cause -----
    scores = {key: 0 for key in cause_to_needs.keys()}

    # 1) dandruff / scalp issues
    if "dandruff" in scale_type or "flake" in scale_type or "itch" in scale_type:
        scores["Dandruff_Fungal"] += 3
        scores["Seborrheic_Dermatitis"] += 2
    if "dry" in scale_type:
        scores["Dry_Scalp"] += 2
    if "oily" in scale_type:
        scores["Oily_Scalp"] += 2
    if "yes" in anti_dandruff:
        scores["Dandruff_Fungal"] += 2

    # 2) hard water
    if "hard" in water_type:
        scores["Hard_Water_Damage"] += 3
    if "yes" in water_changed and "yes" in hair_loss_increased:
        scores["Hard_Water_Damage"] += 2

    # 3) vitamin / mineral stuff
    if "yes" in vit_d:
        scores["Vitamin_D_Deficiency"] += 4
    if "yes" in iron_def:
        scores["Iron_Deficiency"] += 4
        scores["Anemia"] += 2

    # 4) thyroid / hormonal
    if "yes" in thyroid:
        scores["Thyroid_Disorders"] += 4
    if "pregnancy" in recent_illness or "delivery" in recent_illness:
        scores["Postpartum"] += 4

    # 5) stress / sleep
    if (not np.isnan(stress_val) and stress_val >= 7) or (
        not np.isnan(sleep_val) and sleep_val <= 5
    ):
        scores["Stress_Shedding"] += 4
        scores["Stress_Sleep"] += 2

    # 6) lifestyle + diet
    if "more processed" in diet_change or "junk" in diet_change:
        scores["Junk_Food_High_Sugar"] += 3
        scores["Lifestyle_Diet_Change"] += 2
    if not np.isnan(water_val) and water_val < 1.0:
        scores["Low_Water_Intake"] += 3

    # 7) styling / chemical
    if "often" in hair_color:
        scores["Coloring_Bleaching"] += 3
        scores["Chemical_Treatments"] += 2
    if "3+" in heat_tools:
        scores["Heat_Styling"] += 3
    elif "1–2" in heat_tools:
        scores["Heat_Styling"] += 1

    if "yes" in styling_prod:
        scores["Chemical_Treatments"] += 1

    if "yes" in hijab:
        scores["Tight_Hairstyles"] += 2

    # 8) pattern → genetic vs diffuse
    if "front" in pattern or "temple" in pattern:
        scores["Genetic_Androgenetic"] += 3
    if "overall" in pattern or "shedding" in pattern:
        scores["Seasonal_Shedding"] += 1

    # 9) severity
    try:
        sev = float(answers.get("Hair Loss Severity (0–10)", 0))
    except Exception:
        sev = 0.0
    if sev >= 7:
        # big severity + genetic-like pattern
        if scores["Genetic_Androgenetic"] > 0:
            scores["Genetic_Androgenetic"] += 2

    # pick the highest-scoring cause
    best_cause = max(scores.items(), key=lambda x: x[1])[0]
    best_score = scores[best_cause]

    # if everything is zero, call it Mixed_Other
    if best_score == 0:
        return "Mixed_Other"

    return best_cause


# ----------------------------------------------------------
# 5. SESSION STATE INIT
# ----------------------------------------------------------
if "current_section" not in st.session_state:
    st.session_state.current_section = "Hair Loss Assessment"

if "model_cause" not in st.session_state:
    st.session_state.model_cause = "Unknown"

if "food_tips" not in st.session_state:
    st.session_state.food_tips = []

if "supp_tips" not in st.session_state:
    st.session_state.supp_tips = []

if "product_tips" not in st.session_state:
    st.session_state.product_tips = pd.DataFrame()

if "meal_score" not in st.session_state:
    st.session_state.meal_score = None
if "meal_verdict" not in st.session_state:
    st.session_state.meal_verdict = None

if "chronic_summary" not in st.session_state:
    st.session_state.chronic_summary = None

# ----------------------------------------------------------
# 6. LOAD SHARED DATA
# ----------------------------------------------------------
model = load_model()
products_df = load_products()
food_df = load_food_recs()
supp_df = load_supp_recs()

# ----------------------------------------------------------
# 7. SIDEBAR – LOGO + NAV + STEPS
# ----------------------------------------------------------
with st.sidebar:
    if os.path.exists(IMAGE_AROHI_PATH):
        st.image(IMAGE_AROHI_PATH, caption="Arohi Assistant", use_column_width=True)
    else:
        st.markdown("### Arohi Assistant")

    st.markdown("---")
    st.markdown('<span class="sahh-pill">Prototype</span>', unsafe_allow_html=True)
    st.markdown("")

    sections = [
        "Hair Loss Assessment",
        "Care Recommendations",
        "Meal Quality Check",
        "Chronic Check (Optional)",
        "Summary",
    ]

    current = st.session_state.current_section
    idx = sections.index(current) if current in sections else 0

    st.markdown('<p class="sahh-section-label">Section</p>', unsafe_allow_html=True)
    chosen = st.radio("", sections, index=idx)
    st.session_state.current_section = chosen

    st.markdown("---")
    st.markdown(
        """
        <div class="sahh-steps">
        Step 1: Assessment • Step 2: Care tips • Step 3: Meals •
        Step 4: Optional chronic check • Step 5: Summary
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------
# 8. MAIN TITLE + DISCLAIMER
# ----------------------------------------------------------
st.title("Arohi Hair Health Assistant")

st.markdown(
    """
    <div class="sahh-card sahh-note-yellow">
    <strong>Medical note</strong><br>
    Info only. Not a diagnosis and not a replacement for lab tests or doctors.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

# ==========================================================
# SECTION 1 – HAIR LOSS ASSESSMENT
# ==========================================================
if st.session_state.current_section == "Hair Loss Assessment":
    st.header("1. Hair Loss Assessment")

    if model is None:
        st.error("Model file not found or could not be loaded.")
    else:
        with st.form("assessment_form"):
            st.subheader("A. Basics & environment")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 15, 80, 25)
            with c2:
                gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            with c3:
                years_usa = st.selectbox(
                    "Years in USA", ["<1 year", "1–2 years", "2–3 years", "3+ years"]
                )

            c4, c5, c6 = st.columns(3)
            with c4:
                water_type = st.selectbox(
                    "Water type at home", ["Soft", "Hard", "Not sure"]
                )
            with c5:
                water_change = st.selectbox(
                    "Water type changed after moving?",
                    ["Yes", "No", "Not sure"],
                )
            with c6:
                diet_change = st.selectbox(
                    "Diet after moving",
                    [
                        "More protein",
                        "Less protein",
                        "More processed food",
                        "Same as before",
                    ],
                )

            c7, c8, c9 = st.columns(3)
            with c7:
                water_intake = st.selectbox(
                    "Daily water (L)", ["<1L", "1–2L", "2–3L", ">3L"]
                )
            with c8:
                stress = st.slider("Stress (0–10)", 0, 10, 5)
            with c9:
                sleep = st.slider("Sleep hours", 3, 12, 7)

            st.subheader("B. Hair & health")
            c10, c11, c12 = st.columns(3)
            with c10:
                hair_type = st.selectbox(
                    "Hair type", ["Straight", "Wavy", "Curly", "Coily", "Other"]
                )
            with c11:
                scalp_type = st.selectbox(
                    "Scalp type", ["Normal", "Oily", "Dry / flaky", "Sensitive"]
                )
            with c12:
                wash_freq = st.selectbox("Wash / week", ["1", "2", "3", "4+"])

            c13, c14, c15 = st.columns(3)
            with c13:
                oil_hair = st.selectbox("Oil hair?", ["Yes", "No", "Sometimes"])
            with c14:
                heat_tools = st.selectbox(
                    "Heat tools", ["Rarely / never", "1–2 / week", "3+ / week"]
                )
            with c15:
                color_bleach = st.selectbox(
                    "Colour / bleach (6–12 months)",
                    ["No", "Sometimes", "Often"],
                )

            c16, c17, c18 = st.columns(3)
            with c16:
                iron_hist = st.selectbox(
                    "Iron deficiency history", ["No", "Yes", "Not sure"]
                )
            with c17:
                vitd_hist = st.selectbox(
                    "Vitamin D deficiency", ["No", "Yes", "Not sure"]
                )
            with c18:
                thyroid_issue = st.selectbox(
                    "Thyroid issue", ["No", "Yes", "Not sure"]
                )

            c19, c20 = st.columns(2)
            with c19:
                recent_illness = st.selectbox(
                    "Recent major illness",
                    [
                        "No",
                        "COVID in last year",
                        "Surgery in last year",
                        "Pregnancy / delivery",
                        "Other major illness",
                    ],
                )
            with c20:
                worse_after_move = st.selectbox(
                    "Hair loss worse after moving?", ["Yes", "No"]
                )

            st.subheader("D. Hair-loss pattern")
            c21, c22 = st.columns(2)
            with c21:
                severity = st.slider("Hair-loss severity (0–10)", 0, 10, 5)
            with c22:
                pattern = st.selectbox(
                    "Pattern you notice",
                    [
                        "Overall thinning",
                        "Front / temples",
                        "Patchy",
                        "Shedding all over",
                    ],
                )

            submitted = st.form_submit_button("Run assessment")

            if submitted:
                try:
                    # ---- Map water intake choice to numeric liters for the model ----
                    water_intake_map = {
                        "<1L": 0.5,
                        "1–2L": 1.5,
                        "2–3L": 2.5,
                        ">3L": 3.5,
                    }
                    water_intake_num = water_intake_map.get(water_intake, None)

                    # Map UI answers to training feature columns
                    answers = {
                        "Age": age,
                        "Gender": gender,
                        "Country of Origin": None,
                        "State in USA": None,
                        "Years in USA": years_usa,
                        "Water Type at Home": water_type,
                        "Water Type Changed After Coming to USA": water_change,
                        "Diet Change After Moving": diet_change,
                        "Daily Water Intake (Liters)": water_intake_num,
                        "Stress Level (0-10)": stress,
                        "Sleep Hours": sleep,
                        "Hair Type": hair_type,
                        "Scale Type": scalp_type,
                        "Wash Frequency Per Week": wash_freq,
                        "Do You Oil Your Hair?": oil_hair,
                        "Heat Tools Usage": heat_tools,
                        "Hair Color / Bleach": color_bleach,
                        "Hijab or Turban Use": None,
                        "Use of Hair Styling Products": None,
                        "Iron Deficiency": iron_hist,
                        "Vitamin D Deficiency": vitd_hist,
                        "Thyroid Issues": thyroid_issue,
                        "Recent Major Illness": recent_illness,
                        "Hair Loss Severity (0–10)": severity,
                        "Hair Loss Increased After Coming to USA": worse_after_move,
                        "Hair Loss Pattern": pattern,
                        "When Did Hair Loss Start?": None,
                        "Anti-Dandruff Shampoo Use": None,
                        "Top 3 Hair Products You Use": None,
                        "Budget for Hair Products": None,
                        "Consent to Use Data for Academic Research": None,
                    }

                    # ---- DETERMINE CAUSE (USE RULE-BASED QUIZ ENGINE ONLY) ----
                    cause = infer_cause_from_answers(answers)

                    # store final cause
                    st.session_state.model_cause = cause

                    # now pull all recommendations based on this cause
                    st.session_state.food_tips = get_foods_for_cause(cause, food_df, 6)
                    st.session_state.supp_tips = get_supps_for_cause(cause, supp_df, 6)
                    st.session_state.product_tips = get_products_for_cause(
                        cause, products_df, budget_band="low", source=None, top_n=8
                    )

                    st.markdown(
                        '<div class="sahh-card sahh-card-green">Assessment done. Moving to care tips...</div>',
                        unsafe_allow_html=True,
                    )

                    st.session_state.current_section = "Care Recommendations"
                    st.rerun()

                except Exception as e:
                    st.markdown(
                        '<div class="sahh-card sahh-card-pink">Assessment error. Please check the model and CSV columns.</div>',
                        unsafe_allow_html=True,
                    )
                    st.text(str(e))


# ==========================================================
# SECTION 2 – CARE RECOMMENDATIONS
# ==========================================================
elif st.session_state.current_section == "Care Recommendations":
    st.header("2. Care Recommendations")

    cause = st.session_state.model_cause or "Unknown"
    explanation = CAUSE_EXPLANATIONS.get(
        cause, "Your pattern looks mixed – consider multiple lifestyle and health factors."
    )

    st.markdown(
        f"""
        <div class="sahh-card sahh-card-blue">
            <strong>Main hair-loss pattern (model guess):</strong>
            &nbsp;<span class="sahh-cause-tag">{cause}</span><br><br>
            {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### A. Food focus")
    food_tips = st.session_state.food_tips or []
    if not food_tips:
        st.write("No food tips saved yet. You can still focus on protein, iron and veggies.")
    else:
        for tip in food_tips:
            st.markdown(f"- {tip}")
    st.caption("Use these as ideas to discuss with a dietitian, not strict rules.")

    st.markdown("### B. Supplements to discuss")
    supp_tips = st.session_state.supp_tips or []
    if not supp_tips:
        st.write("No supplement tips saved from the file.")
    else:
        for tip in supp_tips:
            st.markdown(f"- {tip}")
    st.caption("Always check dose, interactions and lab tests with a doctor.")

    st.markdown("### C. Hair products")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.selectbox("Budget", ["low", "medium", "high"])
    with col2:
        source = st.selectbox("Shop", ["Any", "Amazon", "Sephora", "Other"])
        source_val = None if source == "Any" else source
    with col3:
        top_n = st.slider("How many suggestions?", 3, 15, 6)

    prod_df = get_products_for_cause(
        cause, products_df, budget_band=budget, source=source_val, top_n=top_n
    )
    st.session_state.product_tips = prod_df

    if prod_df.empty:
        st.write("No hair products matched this filter. Try another budget or shop.")
    else:
        st.dataframe(prod_df, use_container_width=True)

    st.caption("These are examples only. Not sponsored and not guaranteed to work.")

    st.markdown("---")
    if st.button("Next: Meal Quality Check"):
        st.session_state.current_section = "Meal Quality Check"
        st.rerun()

# ==========================================================
# SECTION 3 – MEAL QUALITY CHECK
# ==========================================================
elif st.session_state.current_section == "Meal Quality Check":
    st.header("3. Meal Quality Check")

    st.markdown(
        """
        Pick the main food groups in each meal.  
        Aim for **protein + veggies + whole grain** most days.
        """
    )

    meal_names = ["Breakfast", "Lunch", "Dinner"]
    group_options = [
        "**Protein** (Chicken, Fish, Eggs, Paneer, Tofu, Lentils, Chole, Rajma, Sprouts, Greek Yogurt)",
        "**Veg & Fruits** (Spinach, Methi, Lauki, Carrot, Beetroot, Broccoli, Sweet Potato, Banana, Apple, Pomegranate, Mango)",
        "**Whole Grains** (Roti, Brown Rice, Oats, Quinoa, Jowar, Bajra, Ragi, Dalia, Idli/Dosa batter)",
        "**Healthy Fats & Nuts** (Avocado, Olive Oil, Ghee, Almonds, Walnuts, Pistachios, Peanuts, Coconut, Flaxseed, Chia, Sesame)",
        "**Fluids** (Water, Herbal Tea, Coconut Water, Buttermilk, Lemon Water, Soups)",
        "**Fermented Foods** (Yogurt, Curd Rice, Idli, Dosa, Dhokla, Kombucha)",
        "**Sugary / Processed** (Mithai, Pastries, Biscuits, Maggi, Chips, Sodas, Fast Food)",
    ]

    meal_groups = []

    with st.form("meals_form"):
        for meal in meal_names:
            st.subheader(meal)
            st.text_input(f"{meal} example (optional)", key=f"ex_{meal}")
            gs = st.multiselect(
                f"Food groups in {meal.lower()}",
                group_options,
                key=f"grp_{meal}",
            )
            meal_groups.append(gs)

        submitted = st.form_submit_button("Save meal score")

    if submitted:
        total, verdict = evaluate_daily_meals(meal_groups)
        st.session_state.meal_score = total
        st.session_state.meal_verdict = verdict

        st.markdown(
            f"""
            <div class="sahh-card sahh-card-blue">
                <strong>Meal score:</strong> {total:.1f} – {verdict}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.current_section = "Chronic Check (Optional)"
        st.rerun()

# ==========================================================
# SECTION 4 – CHRONIC CHECK (OPTIONAL)
# ==========================================================
elif st.session_state.current_section == "Chronic Check (Optional)":
    st.header("4. Chronic Check (Optional)")

    st.markdown(
        """
        This step is optional. It looks for patterns linked to **PCOS, thyroid or autoimmune** issues.  
        It does **not** diagnose anything.
        """
    )

    with st.form("chronic_form"):
        st.subheader("A. PCOS / hormone type")
        c1, c2, c3 = st.columns(3)
        with c1:
            irregular_periods = st.checkbox("Cycles often irregular")
        with c2:
            heavy_light = st.checkbox("Very heavy or very light bleed")
        with c3:
            acne_hair = st.checkbox("Facial hair / hormonal acne")

        c4, c5 = st.columns(2)
        with c4:
            pcos_weight = st.checkbox("Weight gain around tummy")
        with c5:
            fam_pcos = st.checkbox("Family history of PCOS / fertility issues")

        st.subheader("B. Thyroid type")
        t1, t2, t3 = st.columns(3)
        with t1:
            unexp_weight = st.checkbox("Unexplained weight gain or loss")
        with t2:
            temp_intol = st.checkbox("Very cold or heat-intolerant")
        with t3:
            fatigue = st.checkbox("Ongoing fatigue")

        t4, t5 = st.columns(2)
        with t4:
            palp_anx = st.checkbox("Palpitations / tremor / anxiety")
        with t5:
            fam_thy = st.checkbox("Family thyroid history")

        st.subheader("C. Autoimmune type")
        a1, a2, a3 = st.columns(3)
        with a1:
            joint_pain = st.checkbox("Joint pain with stiffness")
        with a2:
            sun_rash = st.checkbox("Rash worse in sun")
        with a3:
            mouth_ulcers = st.checkbox("Repeat mouth ulcers")

        a4, a5 = st.columns(2)
        with a4:
            fever = st.checkbox("Low-grade fevers")
        with a5:
            fam_auto = st.checkbox("Family autoimmune disease")

        submit_flags = st.form_submit_button("Review risk pattern")

    if submit_flags:
        flags = {
            "irregular_periods": irregular_periods,
            "heavy_light": heavy_light,
            "acne_hair": acne_hair,
            "pcos_weight": pcos_weight,
            "fam_pcos": fam_pcos,
            "unexp_weight": unexp_weight,
            "temp_intol": temp_intol,
            "fatigue": fatigue,
            "palp_anx": palp_anx,
            "fam_thy": fam_thy,
            "joint_pain": joint_pain,
            "sun_rash": sun_rash,
            "mouth_ulcers": mouth_ulcers,
            "fever": fever,
            "fam_auto": fam_auto,
        }
        summary = evaluate_chronic_flags(flags)
        st.session_state.chronic_summary = summary

        st.markdown(
            f"""
            <div class="sahh-card sahh-card-pink">
            {summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.current_section = "Summary"
        st.rerun()

# ==========================================================
# SECTION 5 – SUMMARY
# ==========================================================
elif st.session_state.current_section == "Summary":
    st.header("5. Summary")

    st.subheader("Main hair-loss reason")
    cause = st.session_state.model_cause or "Unknown"
    explanation = CAUSE_EXPLANATIONS.get(
        cause, "Your pattern looks mixed – consider multiple lifestyle and health factors."
    )
    st.markdown(
        f"""
        <div class="sahh-card">
            Model guess: <span class="sahh-cause-tag">{cause}</span><br><br>
            {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Food focus")
    if st.session_state.food_tips:
        for tip in st.session_state.food_tips:
            st.markdown(f"- {tip}")
    else:
        st.write("No food tips saved.")

    st.subheader("Supplements to discuss")
    if st.session_state.supp_tips:
        for tip in st.session_state.supp_tips:
            st.markdown(f"- {tip}")
    else:
        st.write("No supplement tips saved.")

    st.subheader("Hair products")
    prod_df = st.session_state.product_tips
    if isinstance(prod_df, pd.DataFrame) and not prod_df.empty:
        st.dataframe(prod_df, use_container_width=True)
    else:
        st.write("No hair products saved.")

    st.subheader("Meal pattern")
    if st.session_state.meal_score is not None:
        st.write(
            f"Meal score: **{st.session_state.meal_score:.1f}** – {st.session_state.meal_verdict}"
        )
    else:
        st.write("Meal score not saved yet.")

    st.subheader("Chronic health note")
    if st.session_state.chronic_summary:
        st.markdown(
            f"""
            <div class="sahh-card sahh-note-yellow">
            {st.session_state.chronic_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write("Chronic check not filled. It is optional.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Assessment"):
            st.session_state.current_section = "Hair Loss Assessment"
            st.rerun()
    with c2:
        if st.button("Exit site"):
            st.success("You can now close this browser tab.")
