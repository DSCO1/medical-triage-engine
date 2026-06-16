
import sys
import os
sys.path.append(".")
sys.path.append("backend")

from backend.decision_engine import _analyze_case_internal

try:
    print("Testing fever + abcxyz:")
    res = _analyze_case_internal(["fever", "abcxyz"])
    print(res["steps"][0])
    print("-" * 20)
    
    print("Testing itching:")
    res = _analyze_case_internal(["itching"])
    print(res["steps"][0])
    print("-" * 20)
except Exception as e:
    import traceback
    traceback.print_exc()
