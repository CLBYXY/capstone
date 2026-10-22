import os
import json

if __name__ == "__main__":

    # -- Initialisation ------------------------------------------------
    llm, symp_store, exam_store, embeddings = init_agent(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        rebuild_index=True,   # set to True on the first run
    )

    # -- Simulated patient description (one full description with symptoms and examinations) --
    user_input = (
        "I have been feeling very tired for about 3 weeks, and I have lost around 3 kg "
        "of weight over the past 2 months without trying. I have been drinking a lot of "
        "water and urinating frequently, especially at night. I also feel blurry vision "
        "occasionally. My fasting blood glucose this morning was 9.2 mmol/L, and a urine "
        "test showed glucose present in urine."
    )

    print("\n[病人描述]")
    print(user_input)

    # -- Start Agent A directly -------------------------------------
    final_state = run_agent_a(
        user_input    = user_input,
        llm           = llm,
        symptom_store = symp_store,
        exam_store    = exam_store,
        embeddings    = embeddings,
        verbose       = True,
    )

    print("\n[最终 session state]")
    print(json.dumps(final_state.get("history", []), indent=2, ensure_ascii=False))