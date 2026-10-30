import json
import time
from pathlib import Path

from agent import run_agent


def tools_used(messages):
    names = []
    for content in messages:
        for part in content.parts or []:
            if part.function_call:
                names.append(part.function_call.name)
    return names


cases = json.loads(Path("eval_cases.json").read_text(encoding="utf-8"))
passed = 0
errors = 0

for case in cases:
    print(f"\n=== {case['id']} ===")
    messages = []
    try:
        answer = run_agent(case["question"], messages)
    except Exception as e:
        print(f"ERROR: {type(e).__name__} {getattr(e, 'code', '')}")
        errors += 1
        continue

    used = tools_used(messages)
    problems = []

    if "expected_tool" in case and case["expected_tool"] not in used:
        problems.append(f"did not use {case['expected_tool']}")
    if "forbidden_tool" in case and case["forbidden_tool"] in used:
        problems.append(f"used forbidden tool {case['forbidden_tool']}")
    for word in case.get("must_include", []):
        if word.lower() not in answer.lower():
            problems.append(f"answer is missing '{word}'")
    for word in case.get("must_not_include", []):
        if word.lower() in answer.lower():
            problems.append(f"answer contains '{word}'")

    if problems:
        print("FAIL:", "; ".join(problems))
        print("Answer was:", answer)
    else:
        print("PASS")
        passed += 1

    time.sleep(5)

failed = len(cases) - passed - errors
print(f"\nPassed: {passed}   Failed: {failed}   Errors: {errors}   (out of {len(cases)})")