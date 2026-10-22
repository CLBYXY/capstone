import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import zipfile
import faiss
import torch


# =========================
# Common Functions
# =========================
def combine_component_system(row):
    comp = row['COMPONENT']
    sys = row['SYSTEM_CLEAN']

    if pd.isna(sys) or sys in REMOVE_SYSTEM:
        return comp
    else:
        return comp + " " + sys


def clean_system(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()
    x = x.replace('^', '')
    if '>' in x:
        x = x.split('>')[-1]
    if x in SYSTEM_MAP:
        x = SYSTEM_MAP[x]

    return x


def examination_type_pipeline(input_file, loinc_zip):

    # =========================
    # Step1: rule-based cleaning and noise filtering
    # =========================
    df_event = pd.read_csv(input_file, encoding='latin1')
    df_event['examination_type'] = df_event['examination_type'].astype(str).str.lower()

    df_event.loc[df_event['examination_type'] == 'nan', 'examination_type'] = None

    exam_keywords = [
        'test', 'exam', 'check', 'lab',
        'blood', 'urine', 'mri', 'ct', 'scan', 'x-ray', 'ultrasound',
        'monitor', "follow-up"
    ]

    date_pattern = (
        r'\b('
        r'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
        r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|'
        r'oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|'
        r'day|days|week|weeks|month|months|year|years'
        r')\b'
    )

    date_df = df_event[df_event['examination_type'].str.contains(date_pattern, na=False)]

    no_exam_df = date_df[
        ~date_df['examination_type'].str.contains('|'.join(exam_keywords), na=False)
    ]

    df_event.loc[no_exam_df.index, 'examination_type'] = None


    # =========================
    # Step2: LOINC space construction (SYSTEM cleaning + component building)
    # =========================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with zipfile.ZipFile(loinc_zip, 'r') as z:
        with z.open('Loinc_2.82/LoincTable/Loinc.csv') as f:
            df_loinc = pd.read_csv(f, encoding='latin1', low_memory=False)

    df_loinc['SYSTEM'] = df_loinc['SYSTEM'].astype(str).str.lower().str.strip()
    df_loinc['COMPONENT'] = df_loinc['COMPONENT'].astype(str).str.lower().str.strip()

    delete_keywords = [
        'provider','caregiver','person',
        'device','ventilator','pump',
        'organization','population','event',"setting",
        'xxx','isolate','dose',
        'water',
        'facility','telehealth','encounter',
        'treatment plan',
        'medication',
        'unk',"x12^277"
    ]

    keyword_pattern = '|'.join(delete_keywords)
    x_pattern = r'\bx\d+(?:\^\d+)?\b'

    df_loinc['to_delete'] = (
        df_loinc['SYSTEM'].str.contains(keyword_pattern, case=False, na=False) |
        df_loinc['SYSTEM'].str.contains(x_pattern, case=False, na=False)
    )

    vc = df_loinc['SYSTEM'].value_counts()
    top500_systems = vc.head(500).index.tolist()

    df_loinc_filtered = df_loinc[
        (~df_loinc['to_delete']) &
        (df_loinc['SYSTEM'].isin(top500_systems))
    ].copy()

    global SYSTEM_MAP
    SYSTEM_MAP = {
        'bld': 'blood','blda': 'blood','bldv': 'blood','bldc': 'blood','bld/tiss': 'blood',
        'ser': 'serum/plasma','plas': 'serum/plasma','ser/plas': 'serum/plasma',
        'urine': 'urine','tiss': 'tissue',
        'fluid': 'body_fluid','fld': 'body_fluid'
    }

    global REMOVE_SYSTEM
    REMOVE_SYSTEM = set([
        'patient','outpatient','hospital','fetus',
        'report','study','specimen',
        'device','flu.nonbiological','feed','tpn',
        'rbc','gas','gland','member'
    ])

    df_loinc_filtered['SYSTEM_CLEAN'] = df_loinc_filtered['SYSTEM'].apply(clean_system)

    df_loinc_filtered['component_full'] = df_loinc_filtered.apply(
        combine_component_system,
        axis=1
    )

    types = df_event['examination_type'].dropna().unique().tolist()
    components = df_loinc_filtered['component_full'].dropna().unique().tolist()


    # =========================
    # Step3: embedding + FAISS matching
    # =========================
    model = SentenceTransformer(
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        device=device
    )

    def encode_gpu(texts, batch_size=512):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            emb = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(emb)
        return np.vstack(embeddings).astype('float32')

    type_emb = encode_gpu(types)
    comp_emb = encode_gpu(components)

    index = faiss.IndexFlatIP(comp_emb.shape[1])
    index.add(comp_emb)

    scores, indices = index.search(type_emb, 1)

    mapping_df = pd.DataFrame({
        'examination_type': types,
        'best_component': [components[i] for i in indices.flatten()],
        'similarity': scores.flatten()
    })

    mask_nan = (
        mapping_df['examination_type'].isna() |
        (mapping_df['examination_type'].astype(str).str.lower() == 'nan')
    )

    mapping_df.loc[mask_nan, ['examination_type','best_component','similarity']] = None

    mapping_df.to_csv('STEP4-examination_mapping.csv', index=False)


    # =========================
    # Step4: apply mapping and overwrite original type column
    # =========================
    map_dict = dict(zip(
        mapping_df['examination_type'],
        mapping_df['best_component']
    ))

    df_event['examination_type'] = df_event['examination_type'].map(map_dict)

    df_event.to_csv("STEP4-examination_events_cleaned.csv", index=False)