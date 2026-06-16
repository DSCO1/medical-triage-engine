
from backend.decision_engine import analyze_case

def test(symptoms):
    print(f"Testing: {symptoms}")
    res = analyze_case(symptoms)
    print(f"Specialist: {res['specialist']}")
    print(f"Urgency: {res['urgency']}")
    print(f"Steps: {res['steps'][0]}") # Step 1
    for step in res['steps']:
        if "Step 4" in step:
            print(step)
    print("-" * 20)

test(["fever"])
test(["fatigue"])
test(["uneasiness"])
test(["burning urination"])
test(["abcxyz"])
test(["unknown"])
test(["fever", "abcxyz"])
