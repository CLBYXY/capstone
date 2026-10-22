import os

# -- Initialisation ------------------------------------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm, symp_store, exam_store, embeddings = init_agent_b(
    openai_api_key=OPENAI_API_KEY,
    rebuild_index=False,
)

# -- Patient description --------------------------------------------------
test_input = (
    "Hello doctor, I'm a 52-year-old male. Over the past six months, "
    "I've been feeling that something isn't quite right with my body. "
    "The main issue is that I feel very thirsty all the time. I drink a lot of water every day, "
    "probably around 3 to 4 liters. I also wake up several times at night to drink water and go "
    "to the bathroom—usually about three to four times per night. "
    "I've also been urinating very frequently during the day. "
    "Recently, I've been feeling more fatigued than usual. I used to have no problem climbing "
    "three flights of stairs, but now I feel short of breath after just two floors. "
    "I've lost about 5 kilograms over the past few months, even though I've actually been eating "
    "more than before. I often feel hungry and want to eat again shortly after meals. "
    "I've also noticed some blurred vision, especially at night. Sometimes my eyes feel dry as well. "
    "I haven't been diagnosed with diabetes before, but my father has a history of diabetes. "
    "Recently, during a workplace health check-up, I was told that my fasting blood glucose was "
    "elevated (around 7 mmol/L), and I was advised to come to the hospital for further evaluation. "
    "I'm a bit concerned that I might have diabetes, so I came here today to get it checked."
)

# -- Start Agent B ------------------------------------------------
result = run_agent_b(
    user_input=test_input,
    llm=llm,
    symptom_store=symp_store,
    exam_store=exam_store,
    embeddings=embeddings,
    verbose=True,
)

# -- Print the final return structure ---------------------------------------
print("\n[最终返回结构]")
for k, v in result.items():
    if k not in ("t4_result", "history"):
        print(f"  {k}: {v}")

history = result.get("history", [])
if history:
    print(f"  history: {len(history)} 轮")
    for rec in history:
        print(
            f"    轮次{rec['turn']}  mean_score={rec.get('mean_score')}  "
            f"top_status={rec.get('top_status')}  "
            f"has_t4={rec.get('t4_result') is not None}"
        )