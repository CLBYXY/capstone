import os
import json
import time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# 1. Basic configuration
# =========================

API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = "gpt-5-mini" 

INPUT_EXCEL_PATH = "neuroendocrine_tumor_translated.xlsx"
PROMPT_PATH = "prompts/trainee_doctor_prompt_zh.txt"
OUTPUT_EXCEL_PATH = "neuroendocrine_tumor_translated_output.xlsx"

MAX_WORKERS = 90

# =========================
# 2. Initialise the OpenAI client
# =========================

client = OpenAI(api_key=API_KEY)


# =========================
# 3. Read the prompt template
# =========================

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()


# =========================
# 4. Read the data
# =========================

df = pd.read_excel(INPUT_EXCEL_PATH)


# =========================
# 5. Function that calls the model
# =========================

def call_llm(text: str):
    """
    Input: doctor-patient conversation text
    Return the parsed JSON dict
    """
    prompt = prompt_template.replace("{text}", text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        #temperature=0
    )

    content = response.choices[0].message.content.strip()

    # Compatibility: strip the ```json code-block marker the model may return
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        print(f"JSON 解析失败，原始输出：{content[:100]}...")
        return None

    return result


# =========================
# 6. Per-row processing (wrapped for concurrency)
# =========================

def process_single_row(idx, row):
    text = str(row["msg_translated"])
    if not text or text.lower() == "nan":
        return idx, None

    result = call_llm(text)
    return idx, result

# =========================
# 7. Concurrent main loop
# =========================

print(f"开始处理，使用 {MAX_WORKERS} 线程加速...")

# Use ThreadPoolExecutor for concurrent processing
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_idx = {executor.submit(process_single_row, idx, row): idx for idx, row in df.iterrows()}

    for future in as_completed(future_to_idx):
        idx, result = future.result()

        if result:
            # --- 1. Process Symptoms (column 1) & Examination results (column 2) ---
            # make sure the output uses the "1. 2." line-break format
            for col_key, df_col in [("Symptoms", "Symptoms"), ("Examination results", "Examination_results")]:
                data = result.get(col_key, [])
                if isinstance(data, list):
                    # if the model returned a list, make sure every item is numbered and join them with line breaks
                    formatted_items = []
                    for i, item in enumerate(data, 1):
                        item_str = str(item).strip()
                        # if the model did not number the item, add the number
                        if not item_str.startswith(f"{i}."):
                            formatted_items.append(f"{i}. {item_str}")
                        else:
                            formatted_items.append(item_str)
                    df.at[idx, df_col] = "\n".join(formatted_items)
                else:
                    df.at[idx, df_col] = str(data)

            # --- 2. Process Type of illness (column 3) ---
            # extract the content, remove extra JSON symbols, keep the [type][category] format
            illness = result.get("Type of illness", "")
            if isinstance(illness, list) and len(illness) > 0:
                illness = illness[0]  # take the first element of the list
            df.at[idx, "Type_of_illness"] = str(illness).strip()

            # --- 3. Process Confirmed (column 4) ---
            # output yes/no/unknown directly, removing extra symbols
            confirmed = result.get("Confirmed", "")
            if isinstance(confirmed, list) and len(confirmed) > 0:
                confirmed = confirmed[0]
            df.at[idx, "Confirmed"] = str(confirmed).strip().lower()

            print(f"已完成第 {idx} 行处理")

# =========================
# 8. Save the results
# =========================

df.to_excel(OUTPUT_EXCEL_PATH, index=False)
print(f"处理完成，结果已保存至 {OUTPUT_EXCEL_PATH}")