import os

# ─────────────────────────────────────────────
# 12. Usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # uncomment on the first run to build the index
    # build_exam_vector_index()

    init_tool(openai_api_key=os.environ["OPENAI_API_KEY"])

    test_input = (
        "Fasting blood glucose is 8.5 mmol/L, "
        "HbA1c came back at 7.2%, "
        "and urine protein was found to be positive."
    )

    result = exam_matching_pipeline(
        test_input, _llm, _type_vector_store, _embeddings
    )
    print(json.dumps(convert_numpy(result), indent=2, ensure_ascii=False))

    print(retrieve_by_examinations.invoke(test_input))