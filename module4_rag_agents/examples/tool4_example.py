
# ─────────────────────────────────────────────
# 7. Usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":

    tool3_mock = {
        "candidate_diseases": [
            {
                "disease": "diabetes", "final_score": 4.05, "softmax_prob": 0.72,
                "status_distribution": {
                    "further examination required":       0.55,
                    "diagnosed at the time of the visit": 0.35,
                    "no disease":                         0.10,
                },
            },
            {
                "disease": "hypertension", "final_score": 1.80, "softmax_prob": 0.18,
                "status_distribution": {
                    "further examination required":       0.60,
                    "diagnosed at the time of the visit": 0.30,
                    "no disease":                         0.10,
                },
            },
            {
                "disease": "hypothyroidism", "final_score": 1.20, "softmax_prob": 0.10,
                "status_distribution": {
                    "diagnosed at the time of the visit": 0.50,
                    "further examination required":       0.30,
                    "no disease":                         0.20,
                },
            },
        ],
    }

    done_exams = ["fasting blood glucose"]

    print("=== recommend_exams ===")
    result = recommend_exams(tool3_mock["candidate_diseases"], done_exams)
    print(json.dumps(convert_numpy(result), indent=2, ensure_ascii=False))

    print("\n=== @tool 调用 ===")
    print(recommend_further_exams.invoke({
        "tool3_json":      json.dumps(tool3_mock),
        "done_exams_json": json.dumps(done_exams),
    }))