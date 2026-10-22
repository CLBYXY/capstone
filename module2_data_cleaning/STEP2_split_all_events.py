import pandas as pd
import re


# =========================
# Common Functions
# =========================
def split_numbered_items(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    matches = list(re.finditer(r'\d+\.\s', text))
    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        items.append(text[start:end].strip())
    return items


def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'[,\，]\s*$', '', text)
    text = text.rstrip('.')
    return text.strip()


def extract_time(text):
    if pd.isna(text):
        return None
    text = str(text)
    m1 = re.search(r'duration[:：]\s*(.*)', text, re.IGNORECASE)
    if m1:
        return m1.group(1).strip().rstrip('.')
    m2 = re.search(r'last time[:：]\s*(.*)', text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip().rstrip('.')
    return None


def extract_exam_type(text):
    parts = re.split(r'[:：]', text, maxsplit=1)
    if len(parts) == 2:
        return clean_text(parts[0])
    return None


def extract_exam_result(text):
    parts = re.split(r'[:：]', text, maxsplit=1)
    if len(parts) == 2:
        return clean_text(parts[1])
    return None


# =========================
# MAIN FUNCTION (original logic kept)
# =========================
def split_all_events(input_file):

    # =========================
    # Step1: filter
    # =========================
    df = pd.read_csv(input_file)
    df["conversation_id"] = df["S/N"].astype(str).str.extract(r'(\d+)').astype(int)
    ext_df = pd.read_csv("extracted_information.csv")
    ext_df["conversation_id"] = ext_df["conversation_id"].astype(int)

    df["cluster_name"] = df["cluster_name"].astype(str)
    ext_df["cluster_name"] = ext_df["cluster_name"].astype(str)

    df = df.merge(
        ext_df[["conversation_id", "cluster_name", "status"]],
        on=["conversation_id", "cluster_name"],
        how="left"
    )

    df_base = df[
        (df["cluster_name"] == "diabetes") &
        (df["Confirmed"] != "diagnosed before the visit")
    ]

    df_need_filtered = df_base[df_base["status"] == "planned"]


    # =========================
    # Step2: Symptom（df_base）
    # =========================
    symptom_rows = []

    for _, row in df_base.iterrows():
        items = split_numbered_items(row["Symptoms"])
        for item in items:
            symptom = clean_text(item)
            time = extract_time(item)

            symptom_rows.append({
                "cluster_name": row["cluster_name"],
                "conversation_id": row["conversation_id"],
                "symptom": symptom,
                "time": time,
                "confirmed": row["Confirmed"]
            })

    df_symptom = pd.DataFrame(symptom_rows)


    # =========================
    # Step3: Examination（df_base）
    # =========================
    exam_rows = []

    for _, row in df_base.iterrows():
        items = split_numbered_items(row["Examination_results"])
        for item in items:
            exam_type = extract_exam_type(item)
            exam_result = extract_exam_result(item)

            if exam_type is not None and exam_result is not None:
                exam_rows.append({
                    "cluster_name": row["cluster_name"],
                    "conversation_id": row["conversation_id"],
                    "examination_type": exam_type,
                    "examination_result": exam_result,
                    "confirmed": row["Confirmed"]
                })

    df_examination = pd.DataFrame(exam_rows)


    # =========================
    # Step4: Need（df_need_filtered）
    # =========================
    need_rows = []

    for _, row in df_need_filtered.iterrows():
        items = split_numbered_items(row["Further_examination"])
        for item in items:
            need = clean_text(item)

            need_rows.append({
                "cluster_name": row["cluster_name"],
                "conversation_id": row["conversation_id"],
                "test_need": need,
                "confirmed": row["Confirmed"],
                "status": row["status"]
            })

    df_need = pd.DataFrame(need_rows)


    # =========================
    # Step5: OUTPUT
    # =========================
    df_symptom.to_csv("STEP2-symptom_events.csv", index=False)
    df_examination.to_csv("STEP2-examination_events.csv", index=False)
    df_need.to_csv("STEP2-examination_need.csv", index=False)