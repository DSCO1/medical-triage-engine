"""
NabzAI - Decision Engine Terminal Tests
Runs multiple triage scenarios and prints results.

Usage:
    cd backend
    python test_decision.py
"""

import json
import sys
from decision_engine import analyze_case


# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# -------------------------------------------------------------------
# Test Case Definitions
# -------------------------------------------------------------------

TEST_CASES = [
    # -- HIGH URGENCY -----------------------------------------------
    {
        "name": "[HIGH] Chest pain + breathing issue (combo rule)",
        "input": {
            "symptoms": ["chest pain", "breathing issue"],
            "severity": "high",
            "duration_days": 2,
        },
        "expected_urgency": "High",
    },
    {
        "name": "[HIGH] Chest pain alone",
        "input": {
            "symptoms": ["chest pain"],
            "severity": "normal",
            "duration_days": 1,
        },
        "expected_urgency": "High",
    },
    {
        "name": "[HIGH] Breathing difficulty only",
        "input": {
            "symptoms": ["breathing difficulty"],
            "severity": "normal",
            "duration_days": 1,
        },
        "expected_urgency": "High",
    },

    # -- MEDIUM URGENCY ---------------------------------------------
    {
        "name": "[MEDIUM] Fever lasting 5 days",
        "input": {
            "symptoms": ["fever"],
            "severity": "normal",
            "duration_days": 5,
        },
        "expected_urgency": "Medium",
    },
    {
        "name": "[MEDIUM] Stomach pain",
        "input": {
            "symptoms": ["stomach pain"],
            "severity": "normal",
            "duration_days": 2,
        },
        "expected_urgency": "Medium",
    },
    {
        "name": "[MEDIUM] Dizziness",
        "input": {
            "symptoms": ["dizziness"],
            "severity": "normal",
            "duration_days": 3,
        },
        "expected_urgency": "Medium",
    },

    # -- LOW URGENCY ------------------------------------------------
    {
        "name": "[LOW] Headache",
        "input": {
            "symptoms": ["headache"],
            "severity": "low",
            "duration_days": 1,
        },
        "expected_urgency": "Low",
    },
    {
        "name": "[LOW] Mild cold",
        "input": {
            "symptoms": ["mild cold"],
            "severity": "low",
            "duration_days": 2,
        },
        "expected_urgency": "Low",
    },
    {
        "name": "[LOW] Skin rash",
        "input": {
            "symptoms": ["skin rash"],
            "severity": "normal",
            "duration_days": 4,
        },
        "expected_urgency": "Low",
    },

    # -- EDGE CASES -------------------------------------------------
    {
        "name": "[EDGE] Empty symptoms list",
        "input": {
            "symptoms": [],
            "severity": "normal",
            "duration_days": 1,
        },
        "expected_urgency": "Low",
    },
    {
        "name": "[EDGE] Unknown symptom (ML hybrid fallback)",
        "input": {
            "symptoms": ["purple tongue"],
            "severity": "normal",
            "duration_days": 1,
        },
        "expected_urgency": "Low",  # ML model fallback for unknown symptom + normal severity -> Low
    },
    {
        "name": "[EDGE] Severity override (low symptoms + high severity)",
        "input": {
            "symptoms": ["headache"],
            "severity": "high",
            "duration_days": 1,
        },
        "expected_urgency": "High",  # Low -> High via severity override
    },
    {
        "name": "[EDGE] Fever under 3 days (should be Low, not Medium)",
        "input": {
            "symptoms": ["fever"],
            "severity": "normal",
            "duration_days": 1,
        },
        "expected_urgency": "Low",
    },
    {
        "name": "[EDGE] Multiple medium symptoms -> escalate to High",
        "input": {
            "symptoms": ["stomach pain", "dizziness"],
            "severity": "normal",
            "duration_days": 4,
        },
        "expected_urgency": "High",  # two medium symptoms escalate
    },
    {
        "name": "[EDGE] Mixed high and low symptoms",
        "input": {
            "symptoms": ["chest pain", "headache", "mild cold"],
            "severity": "normal",
            "duration_days": 2,
        },
        "expected_urgency": "High",
    },

    # -- ML HYBRID ENGINE TESTS ----------------------------------------
    {
        "name": "[ML] Dataset-trained symptoms (itching + skin rash)",
        "input": {
            "symptoms": ["itching", "nodal skin eruptions"],
            "severity": "normal",
            "duration_days": 3,
        },
        "expected_urgency": "Low",  # Dermatology → Low
    },
    {
        "name": "[ML] Keyword inference (depression + anxiety)",
        "input": {
            "symptoms": ["depression", "anxiety"],
            "severity": "normal",
            "duration_days": 10,
        },
        "expected_urgency": "Medium",  # Psychiatry → Medium
    },
    {
        "name": "[ML] Severity override on ML result",
        "input": {
            "symptoms": ["yellowish skin", "dark urine"],
            "severity": "high",
            "duration_days": 5,
        },
        "expected_urgency": "High",  # Dermatology Low + severity high → High
    },
    {
        "name": "[ML] Multi-symptom ML (continuous sneezing + chills)",
        "input": {
            "symptoms": ["continuous sneezing", "chills", "shivering"],
            "severity": "normal",
            "duration_days": 2,
        },
        "expected_urgency": "Low",  # Unknown pattern + normal severity -> Low
    },
    {
        "name": "[STABILITY] Long-duration vague symptoms (uneasiness + 10 days)",
        "input": {
            "symptoms": ["uneasiness"],
            "severity": "normal",
            "duration_days": 10,
        },
        "expected_urgency": "Medium",
    },
    {
        "name": "[STABILITY] Long-duration unknown symptom (purple tongue + 10 days)",
        "input": {
            "symptoms": ["purple tongue"],
            "severity": "normal",
            "duration_days": 10,
        },
        "expected_urgency": "Medium",
    },
]


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------

def run_tests():
    print("=" * 65)
    print("  NabzAI - Decision Engine Test Suite")
    print("=" * 65)

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_CASES, start=1):
        result = analyze_case(**tc["input"])
        match = result["urgency"] == tc["expected_urgency"]

        status = "PASS" if match else "FAIL"
        marker = "[OK]" if match else "[XX]"
        if match:
            passed += 1
        else:
            failed += 1

        print(f"\n-- Test {i}: {tc['name']}")
        print(f"   Input     : {json.dumps(tc['input'], indent=None)}")
        print(f"   Expected  : urgency={tc['expected_urgency']}")
        print(f"   Got       : urgency={result['urgency']}, "
              f"specialist={result['specialist']}, "
              f"confidence={result['confidence']}")
        print(f"   Reason    : {result['reason']}")
        print(f"   Status    : {marker} {status}")

    print("\n" + "=" * 65)
    print(f"  Results: {passed} passed, {failed} failed, "
          f"{passed + failed} total")
    print("=" * 65)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    raise SystemExit(0 if success else 1)
