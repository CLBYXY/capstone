import pandas as pd
import zipfile
import os

from STEP1_DISEASE_UNI import disease_pipeline
from STEP2_split_all_events import split_all_events
from STEP3_symptom_pipeline import symptom_pipeline
from STEP4_examination_type_pipeline import examination_type_pipeline
from STEP5_examination_need_pipeline import examination_need_pipeline


def load_and_merge_zip(zip_path):

    all_dfs = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in z.namelist():
            if file.endswith('.xlsx'):
                with z.open(file) as f:
                    df = pd.read_excel(f)
                    all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_excel("STEP0-merged_input.xlsx", index=False)

    return "STEP0-merged_input.xlsx"


def main():

    # =========================
    # Step0: load zip and merge
    # =========================
    zip_path = "input_data.zip"
    merged_file = load_and_merge_zip(zip_path)
    print("Step0 finished: merged input created")


    # =========================
    # Step1: disease pipeline
    # =========================
    disease_pipeline(merged_file)
    print("Step1 finished: disease unified")


    # =========================
    # Step2: split events
    # =========================
    split_all_events("STEP1-DISEASE_UNI.csv")
    print("Step2 finished: events split")


    # =========================
    # Step3: symptom pipeline
    # =========================
    symptom_pipeline("STEP2-symptom_events.csv")
    print("Step3 finished: symptom cleaned")


    # =========================
    # Step4: examination type
    # =========================
    examination_type_pipeline(
        "STEP2-examination_events.csv",
        "loinc.zip"
    )
    print("Step4 finished: examination type mapped")


    # =========================
    # Step5: examination need
    # =========================
    examination_need_pipeline(
        "STEP2-examination_need.csv",
        "loinc.zip"
    )
    print("Step5 finished: examination need mapped")


if __name__ == "__main__":
    main()