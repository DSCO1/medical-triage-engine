"""
NabzAI – ML Engine (Hybrid Layer)

Trains a TF-IDF + Logistic Regression model on the symptoms dataset
to predict medical categories for symptom combinations that the
rule-based engine cannot handle.

This module is imported by decision_engine.py and used as a fallback
when no critical rules match.
"""

import os
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────
# Disease → Category (Manual Core Mapping)
# ─────────────────────────────────────────────────────────────

DISEASE_TO_CATEGORY = {
    "Fungal infection": "Dermatology",
    "Allergy": "Respiratory",
    "GERD": "Gastro",
    "Peptic ulcer diseae": "Gastro",          # dataset spelling
    "Peptic ulcer disease": "Gastro",
    "Chronic cholestasis": "Gastro",
    "Drug Reaction": "Dermatology",
    "Migraine": "Neurological",
    "Hypertension": "Cardiac",
    "Hypertension ": "Cardiac",               # dataset trailing space
    "Heart attack": "Cardiac",
    "Diabetes": "Endocrine",
    "Diabetes ": "Endocrine",                 # dataset trailing space
    "Hypothyroidism": "Endocrine",
    "Hyperthyroidism": "Endocrine",
    "Hypoglycemia": "Endocrine",
    "Urinary tract infection": "Urology",
    "hepatitis A": "Gastro",
    "Hepatitis B": "Gastro",
    "Hepatitis C": "Gastro",
    "Hepatitis D": "Gastro",
    "Hepatitis E": "Gastro",
    "Alcoholic hepatitis": "Gastro",
    "Jaundice": "Gastro",
    "Gastroenteritis": "Gastro",
    "Malaria": "Infectious",
    "Dengue": "Infectious",
    "Typhoid": "Infectious",
    "Chicken pox": "Infectious",
    "AIDS": "Infectious",
    "Common Cold": "Respiratory",
    "Pneumonia": "Respiratory",
    "Bronchial Asthma": "Respiratory",
    "Tuberculosis": "Respiratory",
    "Cervical spondylosis": "Orthopedic",
    "Osteoarthristis": "Orthopedic",
    "Arthritis": "Orthopedic",
    "Paralysis (brain hemorrhage)": "Neurological",
    "(vertigo) Paroymsal  Positional Vertigo": "Neurological",
    "Varicose veins": "Cardiac",
    "Dimorphic hemmorhoids(piles)": "Gastro",
    "Acne": "Dermatology",
    "Psoriasis": "Dermatology",
    "Impetigo": "Dermatology",
}


# ─────────────────────────────────────────────────────────────
# Scalable Category Keyword System
# ─────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "Dermatology": [
        "skin", "rash", "fungal", "acne", "hair", "itching",
        "pimple", "blister", "nodal", "dischromic", "psoriasis",
    ],
    "Cardiac": [
        "heart", "cardiac", "hypertension", "chest pain",
        "palpitation", "varicose",
    ],
    "Neurological": [
        "brain", "neuro", "migraine", "paralysis", "dizziness",
        "vertigo", "headache", "seizure", "tremor",
    ],
    "Respiratory": [
        "lung", "asthma", "breathing", "cough", "cold",
        "tuberculosis", "pneumonia", "sneez", "phlegm",
        "bronchial", "congestion",
    ],
    "Gastro": [
        "stomach", "liver", "hepatitis", "jaundice", "ulcer",
        "vomiting", "abdomen", "nausea", "acidity", "bowel",
        "constipation", "diarrhea", "bile",
    ],
    "Endocrine": [
        "diabetes", "thyroid", "hormone", "sugar", "insulin",
        "glucose",
    ],
    "Urology": [
        "kidney", "urine", "urinary", "bladder", "micturition",
    ],
    "Orthopedic": [
        "bone", "joint", "arthritis", "knee", "muscle",
        "spondylosis", "neck pain", "back pain",
    ],
    "Ophthalmology": [
        "eye", "vision", "blurred",
    ],
    "Dental": [
        "tooth", "teeth", "gum",
    ],
    "ENT": [
        "ear", "nose", "throat", "sinus",
    ],
    "Infectious": [
        "fever", "infection", "viral", "bacterial", "malaria",
        "dengue", "typhoid",
    ],
    "Oncology": [
        "tumor", "cancer",
    ],
    "Psychiatry": [
        "depression", "anxiety", "mental", "mood",
    ],
    "General": [],
}


# ─────────────────────────────────────────────────────────────
# Category → Specialist
# ─────────────────────────────────────────────────────────────

CATEGORY_TO_SPECIALIST = {
    "Cardiac": "Cardiologist",
    "Neurological": "Neurologist",
    "Respiratory": "Pulmonologist",
    "Gastro": "Gastroenterologist",
    "Dermatology": "Dermatologist",
    "Endocrine": "Endocrinologist",
    "Urology": "Urologist",
    "Orthopedic": "Orthopedic",
    "Ophthalmology": "Ophthalmologist",
    "Dental": "Dentist",
    "ENT": "ENT Specialist",
    "Infectious": "General Physician",
    "Oncology": "Oncologist",
    "Psychiatry": "Psychiatrist",
    "General": "General Physician",
}

# Category-level urgency defaults
CATEGORY_URGENCY = {
    "Cardiac": "High",
    "Neurological": "Medium",
    "Respiratory": "Medium",
    "Gastro": "Medium",
    "Dermatology": "Low",
    "Endocrine": "Medium",
    "Urology": "Medium",
    "Orthopedic": "Low",
    "Ophthalmology": "Medium",
    "Dental": "Low",
    "ENT": "Low",
    "Infectious": "Medium",
    "Oncology": "High",
    "Psychiatry": "Medium",
    "General": "Low",
}


# ─────────────────────────────────────────────────────────────
# Category Inference via Keywords
# ─────────────────────────────────────────────────────────────

def infer_category(text: str) -> str:
    """Infer a medical category from free text using keyword matching."""
    text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "General"


# ─────────────────────────────────────────────────────────────
# Data Preprocessing & Model Training
# ─────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CSV_PATH = os.path.join(_DATA_DIR, "symptoms_dataset.csv")

# Module-level model state (trained once at import)
_vectorizer: TfidfVectorizer | None = None
_model: LogisticRegression | None = None
_classes: list = []
_model_ready: bool = False


def _train_model():
    """Load the symptoms dataset and train the TF-IDF + LR model."""
    global _vectorizer, _model, _classes, _model_ready

    if not os.path.exists(_CSV_PATH):
        print(f"[ML Engine] WARNING: Dataset not found at {_CSV_PATH}")
        return

    try:
        df = pd.read_csv(_CSV_PATH)

        # Combine all symptom columns into one text string per row
        symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
        df["symptoms_text"] = df[symptom_cols].apply(
            lambda row: " ".join(
                str(v).strip().replace("_", " ").lower()
                for v in row if pd.notna(v) and str(v).strip()
            ),
            axis=1,
        )

        df["disease_clean"] = df["Disease"].str.strip()

        # Assign category: manual map first, keyword inference as fallback
        categories = []
        for _, row in df.iterrows():
            disease = row["disease_clean"]
            combined = row["symptoms_text"] + " " + disease.lower()
            cat = DISEASE_TO_CATEGORY.get(disease)
            if not cat:
                cat = infer_category(combined)
            categories.append(cat)

        df["category"] = categories

        # Train TF-IDF + Logistic Regression
        _vectorizer = TfidfVectorizer(max_features=3000)
        X = _vectorizer.fit_transform(df["symptoms_text"])
        y = df["category"]

        _model = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
        )
        _model.fit(X, y)
        _classes = list(_model.classes_)
        _model_ready = True

        n_cats = y.nunique()
        print(
            f"[ML Engine] Trained on {len(df)} samples, "
            f"{n_cats} categories. Model ready."
        )

    except Exception as exc:
        print(f"[ML Engine] Training failed: {exc}")
        _model_ready = False


# Train on module import
_train_model()


# ─────────────────────────────────────────────────────────────
# Public Prediction API
# ─────────────────────────────────────────────────────────────

def predict_from_symptoms(symptoms_text: str) -> dict:
    """
    Predict the medical category and specialist from free-text symptoms.

    Returns
    -------
    dict
        {
            "category":             str,
            "specialist":           str,
            "secondary_category":   str | None,
            "secondary_specialist": str | None,
            "confidence":           int   (0-100),
            "ml_used":              bool,
            "urgency":              str,
        }
    """
    result = {
        "category": "General",
        "specialist": "General Physician",
        "secondary_category": None,
        "secondary_specialist": None,
        "confidence": 60,
        "ml_used": False,
        "urgency": "Low",
    }

    if not _model_ready or not symptoms_text.strip():
        # Keyword fallback even without ML
        cat = infer_category(symptoms_text)
        result["category"] = cat
        result["specialist"] = CATEGORY_TO_SPECIALIST.get(cat, "General Physician")
        result["urgency"] = CATEGORY_URGENCY.get(cat, "Low")
        return result

    # Vectorize and predict probabilities
    X = _vectorizer.transform([symptoms_text.lower()])
    probas = _model.predict_proba(X)[0]

    # Sort by probability descending
    ranked = sorted(
        zip(_classes, probas), key=lambda x: x[1], reverse=True
    )

    top_category, top_prob = ranked[0]
    second_category, second_prob = ranked[1] if len(ranked) > 1 else (None, 0.0)

    # If ML confidence is too low, fall back to keyword inference
    if top_prob < 0.5:
        keyword_cat = infer_category(symptoms_text)
        if keyword_cat != "General":
            top_category = keyword_cat
            top_prob = max(top_prob, 0.55)  # slight boost for keyword match
        result["ml_used"] = True  # still note ML was consulted
    else:
        result["ml_used"] = True

    result["category"] = top_category
    result["specialist"] = CATEGORY_TO_SPECIALIST.get(
        top_category, "General Physician"
    )
    result["confidence"] = round(top_prob * 100)
    result["urgency"] = CATEGORY_URGENCY.get(top_category, "Low")

    # Secondary specialist: only if second probability >= 0.25
    # AND it is a different category
    if (
        second_category
        and second_prob >= 0.25
        and second_category != top_category
    ):
        result["secondary_category"] = second_category
        result["secondary_specialist"] = CATEGORY_TO_SPECIALIST.get(
            second_category, None
        )

    return result
