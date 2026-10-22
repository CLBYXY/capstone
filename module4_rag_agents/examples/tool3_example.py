# ─────────────────────────────────────────────
# Usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated Tool1 output (symptom matching result)
    tool1_mock = {
        "candidate_diseases": [
            {
                "disease": "diabetes",
                "total_score": 4.3033,
                "softmax_prob": 1.0,
                "status_distribution": {
                    "diagnosed at the time of the visit": 0.782609,
                    "further examination required": 0.086957,
                    "no disease": 0.130435,
                },
                "matched_symptoms": [
                    "Daily fatigue",
                    "Unexplained weight loss",
                    "Polydipsia (increased thirst)",
                    "Frequent urination (polyuria)",
                    "Occasional blurred vision",
                ],
            },
        ],
    }

    # Simulated Tool2 output (examination matching; both examinations fail type_sim>=0.8, no match)
    tool2_mock_no_match = {
        "candidate_diseases": [],
    }

    # Simulated Tool2 output (with matched examinations)
    tool2_mock_with_match = {
        "candidate_diseases": [
            {
                "disease": "diabetes",
                "total_score": 3.8,
                "softmax_prob": 0.85,
                "status_distribution": {
                    "diagnosed at the time of the visit": 0.70,
                    "further examination required": 0.20,
                    "no disease": 0.10,
                },
                "matched_exams": [
                    "Fasting blood glucose:Elevated fasting blood glucose",
                    "HbA1c:HbA1c 7.2%",
                ],
            },
            {
                "disease": "hypothyroidism",
                "total_score": 1.2,
                "softmax_prob": 0.15,
                "status_distribution": {
                    "further examination required": 0.55,
                    "diagnosed at the time of the visit": 0.30,
                    "no disease": 0.15,
                },
                "matched_exams": [
                    "Thyroid function test:TSH elevated",
                ],
            },
        ],
    }

    print("=" * 60)
    print("Case 1: Tool2 无检查匹配（仅症状）")
    print("=" * 60)
    r1 = combine_results(tool1_mock, tool2_mock_no_match)
    print(json.dumps(convert_numpy(r1), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Case 2: Tool1 + Tool2 均有匹配")
    print("=" * 60)
    r2 = combine_results(tool1_mock, tool2_mock_with_match)
    print(json.dumps(convert_numpy(r2), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Case 3: 通过 @tool 接口调用（JSON字符串）")
    print("=" * 60)
    import json as _json

    result = combine_diagnosis.invoke({
        "tool1_json": _json.dumps(tool1_mock),
        "tool2_json": _json.dumps(tool2_mock_with_match),
    })
    print(result)