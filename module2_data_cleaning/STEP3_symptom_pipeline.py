import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def symptom_pipeline(input_file):

    # =========================
    # Step1: rule-based cleaning and normalization (Phase1–Phase4)
    # =========================
    df = pd.read_csv(input_file)

    # ---------- Phase1 ----------
    exclude_list_phase1 = [
        "long-standing (patient reports; current bp ~130/70)",
        "since covid-19 (hemoglobin currently >100)",
        "5–6 years; currently taking metformin extended-release and xiaoke pills",
        "5–6 years; current blood pressure reported ~80/120 mmhg",
        "2 years (discovered during recent physical examination)",
        "not specified (prior to recent tcm treatment)"
    ]

    exclude_set_phase1 = set([x.lower() for x in exclude_list_phase1])

    exclude_list_phase2 = [
        "present (duration not specified)"
    ]

    exclude_set_phase2 = set([x.lower() for x in exclude_list_phase2])

    def normalize_time_phase1(text):
        text = str(text).strip()
        text_lower = text.lower()

        if text_lower in exclude_set_phase1:
            return text

        if ('today' in text_lower) or ('yesterday' in text_lower):
            return '1 day'

        if ('current' in text_lower) or ('recent' in text_lower):
            return 'recent'

        if text_lower in exclude_set_phase2:
            return 'recent'

        if (
            ('nan' in text_lower) or
            ('not specified' in text_lower) or
            ('not stated' in text_lower) or
            ('unspecified' in text_lower) or
            ("not provided" in text_lower)
        ):
            return None

        return text

    df['time_cleaned'] = df['time'].apply(normalize_time_phase1)


    # ---------- Phase2 ----------
    word_map = {
        "last year":"1 year","last month":"1 month","last week":"1 week","last day":"1 day",
        "half a month":"0.5 month","half a year":"0.5 year","half a week":"0.5 week","half a day":"0.5 day",
        "a week":"1 week","a day":"1 day","a year":"1 year","a month":"1 month",
        "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"
    }

    sorted_keys = sorted(word_map.keys(), key=lambda x: -len(x))

    def replace_word_numbers(text):
        text = str(text).lower()
        for key in sorted_keys:
            pattern = r'\b' + re.escape(key) + r'\b'
            text = re.sub(pattern, word_map[key], text)
        return text

    word2num = df['time_cleaned'].astype(str).apply(replace_word_numbers)

    pattern_time = r'(\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)\s+(day|days|week|weeks|month|months|year|years)'

    def extract_time(text):
        match = re.search(pattern_time, text)
        if match:
            return match.group(1) + " " + match.group(2).rstrip('s')
        return None

    extracted = word2num.apply(extract_time)

    def normalize_unit(text):
        if pd.isna(text):
            return None

        value = float(re.findall(r'\d+\.?\d*', text)[0])
        unit = text.split()[1]

        if unit == 'day' and value >= 30:
            return f"{int(value // 30)} month"

        if unit == 'month' and value > 12:
            return f"{int(value // 12)} year"

        return text

    normalized = extracted.apply(normalize_unit)

    mask = normalized.notna()
    df.loc[mask, 'time_cleaned'] = normalized[mask]


    # ---------- Phase3 ----------
    def normalize_time_phase3(text):
        text_lower = str(text).lower()

        if 'occasional' in text_lower:
            return 'occasional'

        if 'ongoing' in text_lower:
            return 'ongoing'

        return text

    normalized_phase3 = df['time_cleaned'].astype(str).apply(normalize_time_phase3)
    mask = normalized_phase3.notna()
    df.loc[mask, 'time_cleaned'] = normalized_phase3[mask]


    # ---------- Phase4 ----------
    months = [
        'january','february','march','april','may','june',
        'july','august','september','october','november','december',
        'jan','feb','mar','apr','jun','jul','aug','sep','sept','oct','nov','dec'
    ]

    def normalize_time_phase4(text):
        text_lower = str(text).lower()

        if 'since' in text_lower:
            return None

        if any(month in text_lower for month in months):
            return None

        if 'long-standing' in text_lower or 'long standing' in text_lower:
            return 'long-standing'

        return text

    df['time_cleaned'] = df['time_cleaned'].apply(normalize_time_phase4)


    # =========================
    # Step2: semantic clustering normalization (Phase5)
    # =========================
    df['cleaned_time'] = df['time_cleaned'].astype(str).str.lower()

    filtered = df[
        df['cleaned_time'].notna() &
        (df['cleaned_time'] != 'nan') &
        (df['cleaned_time'] != 'recent') &
        (
            ~df['cleaned_time'].str.contains(
                r'\b\d+\s*(year|years|month|months|week|weeks|day|days)\b',
                case=False,
                na=False
            )
        )
    ].copy()

    texts = filtered['cleaned_time'].tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)

    clustering = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    filtered['cluster'] = labels

    cluster_names = {}
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue

        idx = np.where(labels == label)[0]
        cluster_emb = embeddings[idx]

        centroid = np.mean(cluster_emb, axis=0).reshape(1, -1)
        sims = cosine_similarity(cluster_emb, centroid).flatten()

        best_idx = idx[np.argmax(sims)]
        representative_text = texts[best_idx]

        cluster_names[label] = representative_text

    filtered['cluster_name'] = filtered['cluster'].map(cluster_names)


    # =========================
    # Step3: apply mapping and overwrite original time column
    # =========================
    mask_cluster = filtered['cluster'] != -1
    df.loc[filtered.index[mask_cluster], 'time_cleaned'] = filtered.loc[mask_cluster, 'cluster_name']

    mask_noise = filtered['cluster'] == -1
    df.loc[filtered.index[mask_noise], 'time_cleaned'] = np.nan

    # overwrite original time column
    df['time'] = df['time_cleaned']

    df.to_csv("STEP3-symptom_cleaned.csv", index=False)