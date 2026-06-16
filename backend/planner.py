"""
NabzAI – Follow-Up Planner

Generates structured follow-up plans from triage results.
Used by the Streamlit frontend to present actionable next steps.
"""

from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Plan Templates by Urgency
# ─────────────────────────────────────────────────────────────

_PLAN_TEMPLATES = {
    "High": {
        "priority": "Urgent",
        "timeline": "Within 24 hours",
        "base_actions": [
            "Seek immediate medical attention at the nearest emergency facility",
            "Do not delay – call emergency services if symptoms worsen",
            "Bring a list of current medications and recent medical history",
            "Have someone accompany you to the appointment",
            "Avoid strenuous physical activity until evaluated",
        ],
        "notes": "This is a high-priority case. Immediate specialist evaluation is strongly recommended.",
    },
    "Medium": {
        "priority": "Moderate",
        "timeline": "Within 3–5 days",
        "base_actions": [
            "Schedule an appointment with the recommended specialist",
            "Monitor symptoms and note any changes or worsening",
            "Keep a symptom diary with timestamps for the consultation",
            "Stay hydrated and get adequate rest",
            "Avoid self-medication without professional guidance",
        ],
        "notes": "Timely follow-up is advised. Monitor symptoms and escalate if they worsen.",
    },
    "Low": {
        "priority": "Routine",
        "timeline": "Within 1–2 weeks",
        "base_actions": [
            "Schedule a routine check-up with the recommended specialist",
            "Monitor symptoms at home and note any changes",
            "Maintain a healthy diet and adequate hydration",
            "Get sufficient rest and manage stress levels",
            "Follow up if symptoms persist beyond the recommended timeline",
        ],
        "notes": "Symptoms appear manageable. A routine consultation should suffice.",
    },
}

# Specialist-specific add-on actions
_SPECIALIST_ACTIONS = {
    "Cardiologist": [
        "Avoid caffeine and high-sodium foods until evaluated",
        "Monitor blood pressure if a home monitor is available",
    ],
    "Pulmonologist": [
        "Avoid exposure to smoke, dust, and allergens",
        "Practice controlled breathing exercises if comfortable",
    ],
    "Neurologist": [
        "Avoid bright screens and loud environments if experiencing headaches",
        "Note any triggers that precede symptom episodes",
    ],
    "Gastroenterologist": [
        "Follow a bland diet – avoid spicy, oily, and acidic foods",
        "Note timing of symptoms relative to meals",
    ],
    "Orthopedic": [
        "Avoid heavy lifting or strenuous movement of the affected area",
        "Apply ice to swollen joints for 15-minute intervals",
    ],
    "Dermatologist": [
        "Avoid scratching or irritating affected skin areas",
        "Keep the affected area clean and moisturized",
    ],
    "ENT Specialist": [
        "Gargle with warm salt water if experiencing throat discomfort",
        "Avoid cold beverages and dairy products temporarily",
    ],
    "Emergency Medicine": [
        "Call emergency services immediately (911 / local emergency number)",
        "Do not eat or drink anything until evaluated by a physician",
    ],
    "Emergency": [
        "Call emergency services immediately (911 / local emergency number)",
        "Do not eat or drink anything until evaluated by a physician",
    ],
    "Urologist": [
        "Increase water intake to flush the urinary system",
        "Avoid holding urine for extended periods",
    ],
    "Psychiatrist": [
        "Reach out to a trusted person or mental health helpline if needed",
        "Practice grounding exercises or deep breathing techniques",
    ],
    "General Physician": [
        "Maintain a balanced diet and regular sleep schedule",
    ],
    "Internal Medicine": [
        "Prepare a comprehensive list of all symptoms and their timeline",
    ],
}


def generate_plan(triage_result: dict) -> dict:
    """
    Generate a structured follow-up plan from a triage result.

    Parameters
    ----------
    triage_result : dict
        The output from analyze_case().

    Returns
    -------
    dict
        {
            "title":                str,
            "priority":             str,
            "recommended_actions":  list[str],
            "timeline":             str,
            "notes":                str,
            "generated_at":         str (ISO timestamp),
        }
    """
    urgency = triage_result.get("urgency", "Low")
    specialist = triage_result.get("specialist", "General Physician")
    secondary = triage_result.get("secondary_specialist")
    risk_level = triage_result.get("risk_level", "Mild")

    template = _PLAN_TEMPLATES.get(urgency, _PLAN_TEMPLATES["Low"])

    # Build action list: base + specialist-specific
    actions = list(template["base_actions"])

    specialist_extras = _SPECIALIST_ACTIONS.get(specialist, [])
    actions.extend(specialist_extras)

    if secondary and secondary not in ("None", None):
        sec_extras = _SPECIALIST_ACTIONS.get(secondary, [])
        for action in sec_extras:
            if action not in actions:
                actions.append(action)

    title = f"Follow-Up Plan – {specialist} ({risk_level} Risk)"

    return {
        "title": title,
        "priority": template["priority"],
        "recommended_actions": actions,
        "timeline": template["timeline"],
        "notes": template["notes"],
        "generated_at": datetime.now().isoformat(),
    }
