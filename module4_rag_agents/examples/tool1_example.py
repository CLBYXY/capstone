# ─────────────────────────────────────────────
# 13. Usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    # -- First run: build the index ------------------------------------
    embeddings = _get_embeddings()
    store = build_symptom_vector_index("symptom_faiss_index")

    # -- Normal run: load the index ------------------------------------
    init_tool(OPENAI_API_KEY, index_path="symptom_faiss_index")

    # -- Direct single call of the pipeline ---------------------------------
    llm   = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    emb   = _get_embeddings()
    store = load_symptom_vector_index("symptom_faiss_index")

    user_input = (
        "I have been feeling very tired for about 3 weeks, and I have lost around 3 kg "
        "of weight over the past 2 months without trying. I have been drinking a lot of "
        "water and urinating frequently, especially at night. I also feel blurry vision "
        "occasionally."
    )

    print("\n=== symptom_matching_pipeline ===")
    result = symptom_matching_pipeline(user_input, llm, store, emb)
    print("\n=== candidate_diseases ===")
    print(json.dumps(convert_numpy(result["candidate_diseases"]), indent=2, ensure_ascii=False))

    # -- @tool interface call ------------------------------------------
    print("\n=== @tool 接口 ===")
    print(retrieve_by_symptoms.invoke({"user_symptom_description": user_input}))

    # -- Check the time similarity (exponential decay) ------------------------
    print("\n=== _time_similarity 验证 ===")
    cases = [
        (3, "week",  3, "week",  "完全相同"),
        (3, "week",  4, "week",  "差1周"),
        (3, "week",  6, "week",  "差3周"),
        (3, "week", 12, "week",  "差9周"),
        (2, "month", 2, "month", "完全相同"),
        (2, "month", 5, "month", "差3月"),
        (1, "year",  2, "year",  "差1年"),
    ]
    for iv, iu, kv, ku, label in cases:
        sim = _time_similarity(iv, iu, kv, ku)
        dt  = abs(iv * UNIT_TO_DAYS[iu] - kv * UNIT_TO_DAYS[ku])
        print(f"  {label:12s}  Δt={dt:5.0f}d  sim={sim:.4f}")

    # -- Check the SF-IDF computation -------------------------------------
    print("\n=== SF-IDF 验证（前5个症状节点）===")
    sf_idf = compute_sf_idf()
    for symp_id, scores in list(sf_idf.items())[:5]:
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  symp_id={symp_id[:12]}...  top3={top3}")