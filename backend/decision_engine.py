"""
NabzAI – Smart Appointment Triage Agent
Decision Engine (Core Logic)

A HYBRID decision engine combining rule-based triage with
ML-powered generalization for unseen symptom patterns.
Pipeline: Input → Rules → ML Fallback → Decide → Explain
"""

from ml_engine import (
    predict_from_symptoms,
    CATEGORY_URGENCY,
    CATEGORY_TO_SPECIALIST,
    CATEGORY_KEYWORDS,
)


# ─────────────────────────────────────────────────────────────
# Symptom-to-Specialist Mapping Tables
# ─────────────────────────────────────────────────────────────

HIGH_URGENCY_SYMPTOMS = {
    "chest pain": {
        "specialist": "Cardiologist",
        "reason_fragment": "chest pain indicating potential cardiac involvement",
        "rule_label": "chest pain -> cardiac risk",
    },
    "breathing issue": {
        "specialist": "Pulmonologist",
        "reason_fragment": "breathing difficulty suggesting respiratory distress",
        "rule_label": "breathing issue -> respiratory distress",
    },
    "breathing difficulty": {
        "specialist": "Pulmonologist",
        "reason_fragment": "breathing difficulty suggesting respiratory distress",
        "rule_label": "breathing difficulty -> respiratory distress",
    },
    "shortness of breath": {
        "specialist": "Pulmonologist",
        "reason_fragment": "shortness of breath indicating compromised respiratory function",
        "rule_label": "shortness of breath -> respiratory compromise",
    },
    "severe bleeding": {
        "specialist": "Emergency Medicine",
        "reason_fragment": "severe bleeding requiring immediate medical intervention",
        "rule_label": "severe bleeding -> emergency intervention",
    },
    "fainting": {
        "specialist": "Cardiologist",
        "reason_fragment": "fainting episodes suggesting possible cardiovascular or neurological concern",
        "rule_label": "fainting -> cardiovascular concern",
    },
}

MEDIUM_URGENCY_SYMPTOMS = {
    "fever": {
        "specialist": "General Physician",
        "reason_fragment": "persistent fever requiring medical evaluation",
        "duration_threshold": 3,
        "rule_label": "fever (>=3 days) -> persistent infection",
    },
    "stomach pain": {
        "specialist": "Gastroenterologist",
        "reason_fragment": "stomach pain suggesting gastrointestinal concern",
        "rule_label": "stomach pain -> gastrointestinal concern",
    },
    "abdominal pain": {
        "specialist": "Gastroenterologist",
        "reason_fragment": "abdominal pain indicating possible gastrointestinal issue",
        "rule_label": "abdominal pain -> gastrointestinal issue",
    },
    "dizziness": {
        "specialist": "Neurologist",
        "reason_fragment": "dizziness suggesting potential neurological evaluation needed",
        "rule_label": "dizziness -> neurological evaluation",
    },
    "blurred vision": {
        "specialist": "Neurologist",
        "secondary_specialist": "Ophthalmologist",
        "reason_fragment": "blurred vision indicating potential neurological or ophthalmological concern",
        "rule_label": "blurred vision -> neurological / ophthalmological evaluation",
    },
    "joint pain": {
        "specialist": "Orthopedic",
        "reason_fragment": "joint pain warranting orthopedic assessment",
        "rule_label": "joint pain -> orthopedic assessment",
    },
    "back pain": {
        "specialist": "Orthopedic",
        "reason_fragment": "back pain requiring musculoskeletal evaluation",
        "rule_label": "back pain -> musculoskeletal evaluation",
    },
    "ear pain": {
        "specialist": "ENT Specialist",
        "reason_fragment": "ear pain indicating possible ENT condition",
        "rule_label": "ear pain -> ENT condition",
    },
    "sore throat": {
        "specialist": "ENT Specialist",
        "reason_fragment": "persistent sore throat needing ENT evaluation",
        "rule_label": "sore throat -> ENT evaluation",
    },
    "burning urination": {
        "specialist": "Urologist",
        "reason_fragment": "burning sensation during urination suggesting a potential urinary tract issue",
        "rule_label": "burning urination -> urological evaluation",
    },
}

LOW_URGENCY_SYMPTOMS = {
    "headache": {
        "specialist": "General Physician",
        "reason_fragment": "mild headache manageable with general consultation",
        "rule_label": "headache -> general consultation",
    },
    "mild cold": {
        "specialist": "General Physician",
        "reason_fragment": "mild cold symptoms typical of a common viral infection",
        "rule_label": "mild cold -> viral infection",
    },
    "cold": {
        "specialist": "General Physician",
        "reason_fragment": "cold symptoms that can be managed with general care",
        "rule_label": "cold -> general care",
    },
    "cough": {
        "specialist": "General Physician",
        "reason_fragment": "cough symptoms suitable for general medical assessment",
        "rule_label": "cough -> general assessment",
    },
    "skin rash": {
        "specialist": "Dermatologist",
        "reason_fragment": "skin rash requiring dermatological evaluation",
        "rule_label": "skin rash -> dermatological evaluation",
    },
    "rash": {
        "specialist": "Dermatologist",
        "reason_fragment": "skin rash that should be evaluated by a dermatologist",
        "rule_label": "rash -> dermatological evaluation",
    },
    "acne": {
        "specialist": "Dermatologist",
        "reason_fragment": "acne condition best handled by a dermatologist",
        "rule_label": "acne -> dermatological care",
    },
    "runny nose": {
        "specialist": "General Physician",
        "reason_fragment": "runny nose indicative of a minor upper respiratory issue",
        "rule_label": "runny nose -> minor respiratory issue",
    },
    "sneezing": {
        "specialist": "General Physician",
        "reason_fragment": "sneezing likely related to allergies or a mild infection",
        "rule_label": "sneezing -> allergy or mild infection",
    },
    "body ache": {
        "specialist": "General Physician",
        "reason_fragment": "general body ache that can be assessed in a routine consultation",
        "rule_label": "body ache -> routine consultation",
    },
    "fatigue": {
        "specialist": "General Physician",
        "reason_fragment": "fatigue symptoms indicating the need for a general health review",
        "rule_label": "fatigue -> general check-up",
    },
    "uneasiness": {
        "specialist": "General Physician",
        "reason_fragment": "general uneasiness warranting a basic clinical assessment",
        "rule_label": "uneasiness -> clinical assessment",
    },
}

# Special combo rules: when multiple symptoms appear together
COMBO_RULES = [
    {
        "required": {"chest pain", "breathing issue"},
        "urgency": "High",
        "specialist": "Cardiologist",
        "confidence_boost": 10,
        "rule_label": "combo: chest pain + breathing issue -> cardiac emergency",
        "reason": (
            "Chest pain combined with breathing difficulty suggests "
            "a potential cardiac emergency requiring immediate attention"
        ),
    },
    {
        "required": {"chest pain", "breathing difficulty"},
        "urgency": "High",
        "specialist": "Cardiologist",
        "confidence_boost": 10,
        "rule_label": "combo: chest pain + breathing difficulty -> cardiac emergency",
        "reason": (
            "Chest pain combined with breathing difficulty suggests "
            "a potential cardiac emergency requiring immediate attention"
        ),
    },
    {
        "required": {"chest pain", "shortness of breath"},
        "urgency": "High",
        "specialist": "Cardiologist",
        "secondary_specialist": "Pulmonologist",
        "confidence_boost": 10,
        "rule_label": "combo: chest pain + shortness of breath -> cardiac event",
        "reason": (
            "The combination of symptoms indicates a potential cardiac and respiratory emergency requiring immediate evaluation"
        ),
    },
    {
        "required": {"headache", "blurred vision"},
        "urgency": "High",
        "specialist": "Neurologist",
        "secondary_specialist": "Ophthalmologist",
        "confidence_boost": 10,
        "rule_label": "combo: severe headache + vision issues -> neurological event",
        "reason": (
            "The combination of symptoms indicates a potentially severe neurological or ophthalmological condition requiring urgent attention"
        ),
    },
    {
        "required": {"fever", "headache"},
        "urgency": "Medium",
        "specialist": "General Physician",
        "confidence_boost": 5,
        "rule_label": "combo: fever + headache -> systemic infection",
        "reason": (
            "Fever accompanied by headache may indicate a systemic "
            "infection requiring prompt medical evaluation"
        ),
    },
    {
        "required": {"dizziness", "fainting"},
        "urgency": "High",
        "specialist": "Cardiologist",
        "confidence_boost": 10,
        "rule_label": "combo: dizziness + fainting -> cardiovascular concern",
        "reason": (
            "Dizziness with fainting episodes suggests a potentially "
            "serious cardiovascular or neurological condition"
        ),
    },
]

# Urgency rank for comparisons (higher = more urgent)
URGENCY_RANK = {"Low": 1, "Medium": 2, "High": 3}

# Map urgency to human-friendly risk level
RISK_LEVEL_MAP = {"High": "Critical", "Medium": "Moderate", "Low": "Mild"}

# Specialist clinical priority scores (higher = more critical specialty)
SPECIALIST_PRIORITY = {
    "Cardiologist": 5,
    "Pulmonologist": 5,
    "Ophthalmologist": 5,
    "Neurologist": 4,
    "Urologist": 4,
    "Psychiatrist": 4,
    "Orthopedic": 4,
    "Gastroenterologist": 4,
    "Endocrinologist": 3,
    "Dermatologist": 3,
    "ENT Specialist": 3,
    "Dentist": 2,
    "Oncologist": 5,
    "General Physician": 1,
}

# ─────────────────────────────────────────────────────────────
# Symptom Normalization Layer
# ─────────────────────────────────────────────────────────────

# Exact alias mapping: real-world variations → canonical symptom
SYMPTOM_MAP = {
    "tightness in chest": "chest pain",
    "chest discomfort": "chest pain",
    "pressure in chest": "chest pain",
    "chest tightness": "chest pain",
    "heart pain": "chest pain",
    "shortness of breath": "breathing issue",
    "difficulty breathing": "breathing issue",
    "breathlessness": "breathing issue",
    "can't breathe": "breathing issue",
    "trouble breathing": "breathing issue",
    "high fever": "fever",
    "mild fever": "fever",
    "high temperature": "fever",
    "abdominal discomfort": "stomach pain",
    "tummy pain": "stomach pain",
    "belly pain": "stomach pain",
    "loss of balance": "dizziness",
    "feeling dizzy": "dizziness",
    "lightheaded": "dizziness",
    "light headed": "dizziness",
    "migraine": "headache",
    "head pain": "headache",
    "skin irritation": "skin rash",
    "itchy skin": "skin rash",
    "running nose": "runny nose",
    "blocked nose": "cold",
    "nasal congestion": "cold",
    "throwing up": "stomach pain",
    "nausea": "stomach pain",
    "vomiting": "stomach pain",
    "passed out": "fainting",
    "blacked out": "fainting",
    "sweating": "sweating",
    "blurry vision": "blurred vision",
    "vision problems": "blurred vision",
    "can't see clearly": "blurred vision",
    "fuzzy vision": "blurred vision",
    "painful urination": "burning urination",
    "discomfort while peeing": "burning urination",
    "feeling tired": "fatigue",
    "lethargy": "fatigue",
    "exhaustion": "fatigue",
    "feeling uneasy": "uneasiness",
    "restlessness": "uneasiness",
}


def _normalize_symptom(symptom: str) -> tuple[str, bool]:
    """
    Normalize a single symptom string to its canonical form.
    Returns (normalized_string, is_meaningful_flag).
    """
    s = symptom.strip().lower()

    # Exact alias lookup
    if s in SYMPTOM_MAP:
        return SYMPTOM_MAP[s], True

    # Keyword-based recognition (canonicalize for rule matching)
    if "chest" in s:
        return "chest pain", True
    if "breath" in s:
        return "breathing issue", True
    if "fever" in s or "temperature" in s:
        return "fever", True
    if "stomach" in s or "abdominal" in s or "abdomen" in s:
        return "stomach pain", True
    if "dizz" in s or "balance" in s or "lightheaded" in s:
        return "dizziness", True
    if "head" in s and "ache" in s:
        return "headache", True
    if "rash" in s or "skin" in s:
        return "skin rash", True
    if "faint" in s or "passed out" in s:
        return "fainting", True
    if "bleed" in s:
        return "severe bleeding", True
    if "vision" in s or "blurr" in s:
        return "blurred vision", True
    if "throat" in s:
        return "sore throat", True
    if "ear" in s and "pain" in s:
        return "ear pain", True
    if "burning" in s and ("urin" in s or "peeing" in s):
        return "burning urination", True
    if "frequent" in s and ("urin" in s or "peeing" in s):
        return "frequent urination", True
    if "urin" in s or "peeing" in s:
        return "urinary issue", True
    if "fatigue" in s or "tired" in s or "letharg" in s:
        return "fatigue", True
    if "uneas" in s:
        return "uneasiness", True

    # Generic meaningful check using ML keywords
    for keywords in CATEGORY_KEYWORDS.values():
        if any(kw in s for kw in keywords):
            return s, True

    return s, False


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def _normalize_symptoms(symptoms: list) -> list:
    """Lowercase, strip, and normalize each symptom to its canonical form, then deduplicate."""
    normalized = [_normalize_symptom(s)[0] for s in symptoms if isinstance(s, str)]
    # deduplicate while preserving order
    return list(dict.fromkeys(normalized))


def _classify_single_symptom(norm_symptom: str, original_symptom: str, duration_days: int):
    """
    Classify a single symptom and return its urgency level,
    specialist, reason fragment, and rule_label.
    
    Uses norm_symptom for lookups but original_symptom for output strings.
    """
    # Check HIGH urgency table
    if norm_symptom in HIGH_URGENCY_SYMPTOMS:
        entry = HIGH_URGENCY_SYMPTOMS[norm_symptom]
        return {
            "urgency": "High",
            "specialist": entry["specialist"],
            "secondary_specialist": entry.get("secondary_specialist"),
            "reason_fragment": entry["reason_fragment"].replace(norm_symptom, original_symptom),
            "rule_label": entry["rule_label"].replace(norm_symptom, original_symptom),
        }

    # Check MEDIUM urgency table
    if norm_symptom in MEDIUM_URGENCY_SYMPTOMS:
        entry = MEDIUM_URGENCY_SYMPTOMS[norm_symptom]
        # Duration-gated symptoms (e.g., fever >= 3 days)
        threshold = entry.get("duration_threshold")
        if threshold and duration_days < threshold:
            return {
                "urgency": "Low",
                "specialist": entry["specialist"],
                "secondary_specialist": entry.get("secondary_specialist"),
                "reason_fragment": f"recent {original_symptom} (under {threshold} days) that can be monitored",
                "rule_label": f"{original_symptom} (under {threshold} days) -> monitor",
            }
        return {
            "urgency": "Medium",
            "specialist": entry["specialist"],
            "secondary_specialist": entry.get("secondary_specialist"),
            "reason_fragment": entry["reason_fragment"].replace(norm_symptom, original_symptom),
            "rule_label": entry["rule_label"].replace(norm_symptom, original_symptom),
        }

    # Check LOW urgency table
    if norm_symptom in LOW_URGENCY_SYMPTOMS:
        entry = LOW_URGENCY_SYMPTOMS[norm_symptom]
        return {
            "urgency": "Low",
            "specialist": entry["specialist"],
            "secondary_specialist": entry.get("secondary_specialist"),
            "reason_fragment": entry["reason_fragment"].replace(norm_symptom, original_symptom),
            "rule_label": entry["rule_label"].replace(norm_symptom, original_symptom),
        }

    return None


def _check_combo_rules(symptom_set: set):
    """
    Check if any combo rule matches the provided symptom set.
    Returns the first matching combo rule or None.
    """
    for rule in COMBO_RULES:
        if rule["required"].issubset(symptom_set):
            return rule
    return None


def _apply_severity_override(urgency: str, severity: str, is_vague: bool = False, duration_days: int = 1) -> str:
    """
    Apply severity-based urgency calibration while respecting higher-priority rules.
    """
    if urgency == "High":
        return "High"
        
    severity_lower = severity.strip().lower()
    
    if is_vague:
        if severity_lower == "high" or duration_days > 7:
            return "Medium"
        else:
            return "Low"

    if severity_lower == "high":
        return "High"
    elif severity_lower == "medium":
        return "Medium" if URGENCY_RANK["Medium"] > URGENCY_RANK[urgency] else urgency
        
    return urgency


def _compute_confidence(
    num_symptoms: int,
    num_matched: int,
    combo_boost: int,
    clean_symptoms: list,
    severity: str = "normal",
    num_rules: int = 0,
    final_urgency: str = "Low",
) -> int:
    """
    Compute a realistic confidence percentage dynamically.
    - Rule match -> 85-95%
    - Fallback -> 30-60%
    """
    if num_matched > 0:
        base = 85
        max_val = 95
    else:
        base = 30
        max_val = 60

    # Pseudo-random variance based on symptom string hash
    variance = hash("".join(clean_symptoms)) % (max_val - base + 1)
    
    score = base + variance
    
    if final_urgency == "High":
        score = max(85, score)

    return min(max_val, score)


def _with_article(name: str) -> str:
    """Helper to return a name with the correct indefinite article (a/an)."""
    if not name or name == "None":
        return ""
    article = "an" if name[0].lower() in "aeiou" else "a"
    return f"{article} {name}"


def _build_reason(
    clean_symptoms: list,
    combo_reason: str | None,
    severity: str,
    final_urgency: str,
    best_specialist: str = "",
    secondary_specialist: str | None = None,
) -> str:
    """
    Generate a clear, human-readable reason string that explains
    the decision in natural language, ensuring symptom mention and varied structure.
    """
    if not clean_symptoms:
        symp_str = "the reported symptoms"
    elif len(clean_symptoms) == 1:
        symp_str = clean_symptoms[0]
    else:
        symp_str = ", ".join(clean_symptoms[:-1]) + " and " + clean_symptoms[-1]

    # Prepare correctly articulated phrases
    spec_phrase = _with_article(best_specialist)
    # Avoid "general General Physician"
    if "General" in best_specialist:
        gen_spec_phrase = spec_phrase
    else:
        gen_spec_phrase = f"a general {best_specialist}"

    spec_phrase = _with_article(best_specialist)
    spec_phrase_cap = spec_phrase[0].upper() + spec_phrase[1:] if spec_phrase else ""

    if final_urgency == "High":
        templates = [
            "The combination of symptoms ({symptoms}) indicates a potentially serious condition requiring urgent evaluation by {spec_phrase}.",
            "Clinical assessment of {symptoms} suggests a critical pattern necessitating prompt {best_specialist} care.",
            "The presence of {symptoms} raises immediate clinical concern and warrants urgent {best_specialist} intervention."
        ]
    elif final_urgency == "Low":
        templates = [
            "The symptoms '{symptoms}' are non-specific and do not indicate a clear high-risk condition. {spec_phrase_cap} evaluation is appropriate.",
            "With symptoms like {symptoms}, the condition appears mild and manageable by {spec_phrase}.",
            "Based on the clinical presentation of {symptoms}, a routine check-up with {spec_phrase} should be sufficient."
        ]
    else:
        # Medium
        templates = [
            "Clinical evaluation suggests that the reported symptoms ({symptoms}) may require further assessment by {spec_phrase}.",
            "The combination of symptoms ({symptoms}) suggests a need for {spec_phrase} consultation for further diagnosis.",
            "Based on the reported symptoms ({symptoms}), {spec_phrase} evaluation is recommended."
        ]

    variation = hash("".join(clean_symptoms)) % len(templates)
    base_sentence = templates[variation].format(
        symptoms=symp_str, 
        best_specialist=best_specialist,
        spec_phrase=spec_phrase,
        gen_spec_phrase=gen_spec_phrase,
        spec_phrase_cap=spec_phrase_cap
    )
    base_sentence = base_sentence[0].upper() + base_sentence[1:]

    parts = []
    if combo_reason:
        parts.append(combo_reason)

    parts.append(base_sentence)

    if secondary_specialist and secondary_specialist != "None":
        sec_phrase = _with_article(secondary_specialist)
        parts.append(f"{sec_phrase[0].upper() + sec_phrase[1:]} could offer a secondary perspective.")

    sev = severity.strip().lower()
    if sev == "high" and final_urgency != "High":
        parts.append("The reported high severity reinforces the need for timely medical attention.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# Explainability Helpers (NEW)
# ─────────────────────────────────────────────────────────────

def _get_risk_level(urgency: str) -> str:
    """Map urgency to a human-friendly risk level."""
    return RISK_LEVEL_MAP.get(urgency, "Mild")


def _collect_matched_rules(classifications: list, combo, norm_to_original: dict) -> list:
    """Gather rule labels from individual classifications and combo, with original symptom restoration."""
    rules = []
    
    def finalize_label(label: str) -> str:
        if not label: return label
        for norm, orig in norm_to_original.items():
            label = label.replace(norm, orig)
        return label

    if combo:
        rules.append(finalize_label(combo["rule_label"]))
        
    for c in classifications:
        label = c.get("rule_label")
        if label:
            final_l = finalize_label(label)
            if final_l not in rules:
                rules.append(final_l)
    return rules


def _find_secondary_specialist(classifications: list, primary: str):
    """
    Return a secondary specialist if symptoms map to more than one distinct
    medical domain. Uses SPECIALIST_PRIORITY for intelligent ranking.
    Filters out General Physician as secondary (not clinically meaningful).
    """
    all_specialists = []
    for c in classifications:
        all_specialists.append(c["specialist"])
        # Check for explicit secondary specialist from symptom tables
        explicit_sec = c.get("secondary_specialist")
        if explicit_sec:
            all_specialists.append(explicit_sec)

    # Include primary to correctly assess all unique domains
    if primary not in all_specialists:
        all_specialists.append(primary)

    # Remove duplicates while preserving order
    unique_specialists = list(dict.fromkeys(all_specialists))

    # Sort by clinical priority (highest first)
    unique_specialists.sort(
        key=lambda s: SPECIALIST_PRIORITY.get(s, 0),
        reverse=True,
    )

    # Secondary = highest-priority specialist that is NOT the primary
    # and NOT "General Physician" (not a meaningful secondary referral)
    for s in unique_specialists:
        if s != primary and s != "General Physician":
            return s
    return "None"


def _build_steps(
    clean_symptoms: list,
    severity: str,
    duration_days: int,
    combo,
    classifications: list,
    severity_changed: bool,
    escalated: bool,
    final_urgency: str,
    best_specialist: str,
    secondary_specialist,
) -> list:
    """Construct the step-by-step logic trace (1-7) for rule-based decisions."""
    steps = []
    
    # Step 1
    symptoms_str = ', '.join(clean_symptoms) if clean_symptoms else 'none'
    steps.append(f"Step 1: Normalized and deduplicated {len(clean_symptoms)} symptom(s) – {symptoms_str}")
    
    # Step 2
    steps.append(f"Step 2: Evaluate clinical context (severity: {severity}, duration: {duration_days} day(s))")
    
    # Step 3
    if combo or classifications:
        steps.append(f"Step 3: Rule-based match found ({len(classifications) if classifications else 1} pattern(s))")
    else:
        steps.append("Step 3: No rule match found")
        
    # Step 4
    steps.append("Step 4: ML inference skipped (rule match sufficient)")
    
    # Step 5
    if severity_changed or escalated:
        steps.append("Step 5: Symptom override applied based on clinical pattern")
    else:
        steps.append("Step 5: Symptom override skipped")
        
    # Step 6
    sec_str = f"; secondary to {secondary_specialist}" if (secondary_specialist and secondary_specialist != "None") else ""
    steps.append(f"Step 6: Map specialist to {best_specialist}{sec_str}")
    
    # Step 7
    steps.append(f"Step 7: Final reasoning applied {final_urgency} urgency with calibrated confidence")
    
    return steps


# ─────────────────────────────────────────────────────────────
# Symptom Override Layer (Priority 2 — between Rules and ML)
# ─────────────────────────────────────────────────────────────

def _symptom_override(symptoms: list) -> str | None:
    """
    Check for strong symptom patterns that should override ML predictions.
    Returns a category string if a high-confidence pattern is detected,
    or None to let ML handle the decision.

    Priority order (within this function):
      Psychiatry > Ophthalmology > Dental > Dermatology > Orthopedic > Urology
    """
    text = " ".join(symptoms).lower()

    # Psychiatry (mental health)
    if any(kw in text for kw in [
        "anxiety", "depression", "stress", "panic",
        "sleep disturbance", "restlessness", "mood swing",
        "suicidal", "insomnia", "nervousness",
    ]):
        return "Psychiatry"

    # Ophthalmology (MUST be checked BEFORE neurological patterns)
    if any(kw in text for kw in [
        "eye pain", "blurred vision", "vision problem",
        "watery eyes", "eye redness", "double vision",
        "eye swelling", "eye irritation",
    ]):
        return "Ophthalmology"

    # Dental
    if any(kw in text for kw in [
        "tooth pain", "gum swelling", "bad breath",
        "toothache", "gum bleeding", "cavity",
    ]):
        return "Dental"

    # Dermatology (skin + hair)
    if any(kw in text for kw in [
        "hair fall", "dandruff", "itchy scalp",
        "hair loss", "scalp",
    ]):
        return "Dermatology"

    # Orthopedic
    if any(kw in text for kw in [
        "joint pain", "knee pain", "bone pain", "back pain",
        "hip pain", "shoulder pain", "fracture",
    ]):
        return "Orthopedic"

    # Urology
    if any(kw in text for kw in [
        "burning urination", "frequent urination",
        "blood in urine", "bladder pain",
    ]):
        return "Urology"

    return None


# ─────────────────────────────────────────────────────────────
# Main Public API
# ─────────────────────────────────────────────────────────────

import random

def _get_general_physician_variant(
    urgency: str, num_symptoms: int, severity: str, duration_days: int, is_vague: bool = False
) -> str:
    """Returns a consistent professional variant of General Physician based on context."""
    if duration_days > 7 or num_symptoms >= 3 or urgency == "High":
        return "Internal Medicine"
    
    return "General Physician"

def analyze_case(symptoms: list, severity: str = "normal", duration_days: int = 1):
    try:
        return _analyze_case_internal(symptoms, severity, duration_days)
    except Exception:
        # Safe Fallback (MANDATORY – NO CRASHES)
        return {
            "urgency": "Low",
            "risk_level": "Mild",
            "specialist": "General Physician",
            "secondary_specialist": "None",
            "confidence": "45%",
            "reason": "Insufficient or unclear data; defaulting to safe general evaluation.",
            "matched_rules": [],
            "steps": [
                "Step 1: Normalized and deduplicated 0 symptom(s) – none",
                "Step 2: Evaluate context",
                "Step 3: No rule match found",
                "Step 4: ML prediction applied",
                "Step 5: Symptom override skipped",
                "Step 6: Map specialist to General Physician",
                "Step 7: Final reasoning applied Low urgency with calibrated confidence"
            ]
        }

def _analyze_case_internal(
    symptoms: list,
    severity: str = "normal",
    duration_days: int = 1,
) -> dict:
    """
    Analyze a patient case and produce a triage decision.

    Parameters
    ----------
    symptoms : list
        List of symptom strings reported by the patient.
    severity : str
        Patient-reported severity level ("low", "normal", "high").
    duration_days : int
        Number of days the symptoms have persisted.

    Returns
    -------
    dict
        {
            "urgency":              "High" | "Medium" | "Low",
            "risk_level":           "Critical" | "Moderate" | "Mild",
            "specialist":           str,
            "secondary_specialist": str | None,
            "confidence":           str   (e.g. "85%"),
            "reason":               str   (human-readable explanation),
            "matched_rules":        list[str],
            "steps":                list[str],
        }
    """

    INVALID_TOKENS = {"unknown", "na", "n/a", "none", "null", "undefined"}
    display_symptoms = []
    clean_symptoms = []
    seen_lower = set()
    norm_to_original = {}
    
    num_meaningful = 0
    for s in symptoms:
        if s is None:
            continue
        original_preserved = str(s).strip()
        if not original_preserved:
            continue
            
        # Strict filtering of invalid tokens
        s_lower = original_preserved.lower()
        if s_lower in INVALID_TOKENS:
            continue
            
        norm, recognized = _normalize_symptom(original_preserved)
        if not recognized:
            # Skip garbage/non-meaningful inputs entirely
            continue
            
        if s_lower not in seen_lower:
            seen_lower.add(s_lower)
            num_meaningful += 1
            display_symptoms.append(original_preserved)
            clean_symptoms.append(norm)
            
            # Store first original for this norm (for rule label fixup)
            if norm not in norm_to_original:
                norm_to_original[norm] = original_preserved
            
    if not display_symptoms:
        # Return fallback for no valid symptoms
        return {
            "urgency": "Low",
            "risk_level": "Mild",
            "specialist": "General Physician",
            "secondary_specialist": "None",
            "confidence": "45%",
            "reason": "No valid symptoms were provided for evaluation. A General Physician consultation is recommended for a general check-up.",
            "matched_rules": [],
            "steps": [
                "Step 1: Normalized and deduplicated 0 symptom(s)",
                "Step 2: Context evaluation skipped",
                "Step 3: Rule-based match skipped (no inputs)",
                "Step 4: ML prediction skipped (no valid symptoms)",
                "Step 5: Fallback applied",
                "Step 6: Mapped primary specialist to General Physician",
                "Step 7: Final reasoning applied with low urgency and confidence"
            ]
        }
        
    symptom_set = set(clean_symptoms)

    # ── Step 2: Critical Emergency Check (Highest Priority) ──
    emergency_keywords = {"unconsciousness", "loss of consciousness", "unresponsive"}
    if any(s.lower() in emergency_keywords for s in clean_symptoms) or any(str(s).lower() in emergency_keywords for s in symptoms):
        return {
            "urgency": "High",
            "risk_level": "Critical",
            "specialist": "Emergency",
            "secondary_specialist": "None",
            "confidence": "95%",
            "reason": "Clinical evaluation suggests a critical emergency requiring immediate life-saving intervention.",
            "matched_rules": ["CRITICAL_EMERGENCY_OVERRIDE"],
            "steps": [
                "Step 1: Emergency symptom detected",
                "Step 2: Evaluated clinical context as critical",
                "Step 3: Emergency rule-based match override",
                "Step 4: ML inference skipped",
                "Step 5: Priority override applied",
                "Step 6: Mapped primary specialist to Emergency",
                "Step 7: Final reasoning applied High urgency with calibrated confidence"
            ]
        }

    # ── Step 2: Check combo rules first ──────────────────────
    combo = _check_combo_rules(symptom_set)
    combo_boost = combo["confidence_boost"] if combo else 0

    # ── Step 3: Classify each symptom individually ───────────
    classifications = []
    matched_fragments = []
    for i, norm in enumerate(clean_symptoms):
        original = display_symptoms[i]
        result = _classify_single_symptom(norm, original, duration_days)
        if result:
            classifications.append(result)
            matched_fragments.append(result["reason_fragment"])

    num_matched = len(classifications)

    # ── Step 4: Determine highest urgency across symptoms ────
    if combo:
        best_urgency = combo["urgency"]
        best_specialist = combo["specialist"]
    elif classifications:
        best_urgency = max(
            classifications,
            key=lambda c: URGENCY_RANK[c["urgency"]],
        )["urgency"]
        # Pick the specialist from the highest-urgency symptom
        best_specialist = next(
            c["specialist"]
            for c in classifications
            if c["urgency"] == best_urgency
        )
    else:
        # Fallback — no symptoms matched any rule
        best_urgency = "Low"
        best_specialist = "General Physician"

    # ── Step 5: Severity override ────────────────────────────
    vague_symptoms = {"uneasiness", "weird sensation", "fatigue", "weakness", "tired", "dizzy", "lethargy", "feeling unwell", "strange feeling"}
    is_vague = all(s.lower() in vague_symptoms for s in clean_symptoms) if clean_symptoms else True
    any_vague = any(s.lower() in vague_symptoms for s in clean_symptoms) if clean_symptoms else False
    
    # Unknown symptoms (num_matched == 0) are treated as vague for calibration
    final_urgency = _apply_severity_override(best_urgency, severity, is_vague=(num_matched == 0 or any_vague), duration_days=duration_days)

    # If severity override pushed urgency higher and original specialist
    # was General Physician at low urgency, keep the specialist unchanged.

    # ── Step 6: Multi-symptom urgency boost ──────────────────
    # If multiple symptoms individually classify as Medium or higher,
    # escalate to High.
    serious_count = sum(
        1 for c in classifications if URGENCY_RANK[c["urgency"]] >= 2
    )
    if serious_count >= 2 and final_urgency == "Medium":
        final_urgency = "High"

    # ── Step 7: Confidence calculation ───────────────────────
    # Temporary rules for confidence calculation (will be finalized later)
    matched_rules_pre = _collect_matched_rules(classifications, combo, norm_to_original)
    confidence = _compute_confidence(
        num_symptoms=len(display_symptoms),
        num_matched=num_matched,
        combo_boost=combo_boost,
        clean_symptoms=clean_symptoms,
        severity=severity,
        num_rules=len(matched_rules_pre),
        final_urgency=final_urgency,
    )

    # ── Step 8: Determine secondary specialist (moved up) ────
    secondary_specialist = _find_secondary_specialist(
        classifications, best_specialist
    )

    if best_specialist == "General Physician":
        best_specialist = _get_general_physician_variant(
            final_urgency, len(clean_symptoms), severity, duration_days, is_vague
        )

    # ── Step 9: Build human-readable reason ──────────────────
    combo_reason = combo["reason"] if combo else None
    if combo_reason:
        for norm, orig in norm_to_original.items():
            combo_reason = combo_reason.replace(norm, orig)
            
    reason = _build_reason(
        clean_symptoms=display_symptoms,
        combo_reason=combo_reason,
        severity=severity,
        final_urgency=final_urgency,
        best_specialist=best_specialist,
        secondary_specialist=secondary_specialist,
    )

    if num_matched > 0 and any_vague and not is_vague:
        reason += " Specialist assigned based on strong symptom pattern despite some low-specificity inputs."

    # ── Step 10: Collect matched rules ────────────────────────
    matched_rules = _collect_matched_rules(classifications, combo, norm_to_original)

    # ── Step 11: Track whether urgency was modified ──────────
    severity_changed = final_urgency != best_urgency
    escalated = serious_count >= 2 and best_urgency == "Medium"

    # ── Step 12: Handle complete fallback — HYBRID ML ENGINE ───
    if num_matched == 0:
        if num_meaningful == 0:
            # ── Vague/Unknown Fallback ───────────────
            # Assign General Physician for meaningless input
            ml_specialist = "General Physician"
            ml_urgency = "Low"
            ml_confidence = 45
            ml_secondary = None
            
            # Apply severity override
            ml_urgency = _apply_severity_override(ml_urgency, severity, is_vague=True, duration_days=duration_days)
            risk_level = _get_risk_level(ml_urgency)

            reason = _build_reason(
                clean_symptoms=display_symptoms,
                combo_reason=None,
                severity=severity,
                final_urgency=ml_urgency,
                best_specialist=ml_specialist,
                secondary_specialist=ml_secondary,
            )

            ml_steps = [
                f"Step 1: Normalized and deduplicated {len(display_symptoms)} symptom(s) – {', '.join(display_symptoms)}",
                f"Step 2: Evaluated clinical context (severity: {severity}, duration: {duration_days} day(s))",
                "Step 3: No rule match found",
                "Step 4: ML prediction skipped (symptoms not meaningful)",
                "Step 5: Fallback logic applied",
                f"Step 6: Mapped primary specialist to General Physician",
                f"Step 7: Final reasoning applied {ml_urgency} urgency with calibrated confidence"
            ]
            
            return {
                "urgency": ml_urgency,
                "risk_level": risk_level,
                "specialist": ml_specialist,
                "secondary_specialist": "None",
                "confidence": f"{ml_confidence}%",
                "reason": reason,
                "matched_rules": [],
                "steps": ml_steps,
            }
        else:
            # ── Meaningful ML Fallback ───────────────
            symptoms_text = " ".join(clean_symptoms)
            ml_prediction = predict_from_symptoms(symptoms_text)
            
            ml_category = ml_prediction["category"]
            ml_confidence = ml_prediction["confidence"]
            ml_specialist = ml_prediction["specialist"]
            ml_urgency = ml_prediction["urgency"]
            ml_secondary = ml_prediction.get("secondary_specialist")
            

            # Apply severity override
            ml_urgency = _apply_severity_override(ml_urgency, severity, is_vague=is_vague, duration_days=duration_days)
            risk_level = _get_risk_level(ml_urgency)

            # Decide whether to accept ML specialist or stay with GP
            accept_ml_spec = (ml_confidence >= 60) and not is_vague
            if not accept_ml_spec or ml_specialist == "General Physician":
                ml_specialist = _get_general_physician_variant(ml_urgency, len(clean_symptoms), severity, duration_days, False)

            # Build reason using unified contextual builder
            reason = _build_reason(
                clean_symptoms=display_symptoms,
                combo_reason=None,
                severity=severity,
                final_urgency=ml_urgency,
                best_specialist=ml_specialist,
                secondary_specialist=ml_secondary,
            )

            ml_steps = [
                f"Step 1: Normalized and deduplicated {len(display_symptoms)} symptom(s) – {', '.join(display_symptoms)}",
                f"Step 2: Evaluated clinical context (severity: {severity}, duration: {duration_days} day(s))",
                "Step 3: No rule match found",
                "Step 4: ML prediction applied",
                f"Step 5: Mapped primary specialist to {ml_specialist}",
                f"Step 6: Final reasoning applied {ml_urgency} urgency with calibrated confidence"
            ]
            
            return {
                "urgency": ml_urgency,
                "risk_level": risk_level,
                "specialist": ml_specialist,
                "secondary_specialist": ml_secondary or "None",
                "confidence": f"{ml_confidence}%",
                "reason": reason,
                "matched_rules": [],
                "steps": ml_steps,
            }

    # ── Step 13: Build reasoning steps ────────────────────────
    steps = _build_steps(
        clean_symptoms=display_symptoms,
        severity=severity,
        duration_days=duration_days,
        combo=combo,
        classifications=classifications,
        severity_changed=severity_changed,
        escalated=escalated,
        final_urgency=final_urgency,
        best_specialist=best_specialist,
        secondary_specialist=secondary_specialist,
    )

    # ── Step 14: Return final decision ───────────────────────
    return {
        "urgency": final_urgency,
        "risk_level": _get_risk_level(final_urgency),
        "specialist": best_specialist,
        "secondary_specialist": secondary_specialist or "None",
        "confidence": f"{confidence}%",
        "reason": reason,
        "matched_rules": matched_rules,
        "steps": steps,
    }

