import pandas as pd
import numpy as np
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


def classify_need_system(text):
    if pd.isna(text):
        return None

    text = str(text).lower()

    if any(k in text for k in ['urine', 'ua']):
        return 'urine'
    if any(k in text for k in ['csf', 'cerebrospinal']):
        return 'csf'
    if any(k in text for k in ['stool', 'feces']):
        return 'stool'
    if any(k in text for k in ['saliva']):
        return 'saliva'
    if any(k in text for k in ['sputum']):
        return 'sputum'
    if any(k in text for k in ['tissue', 'biopsy']):
        return 'tissue'

    if any(k in text for k in [
        'blood','serum','plasma',
        'glucose','hemoglobin','hba1c',
        'cholesterol','lipid','insulin'
    ]):
        return 'blood'

    return text


def examination_need_pipeline(input_file, loinc_zip):

    # =========================
    # Step1: build LOINC component space
    # =========================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df_event = pd.read_csv(input_file, encoding='latin1')
    df_event['need'] = df_event['test_need'].astype(str).str.lower().str.strip()

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

    component_df = df_loinc_filtered[['component_full', 'SYSTEM_CLEAN']].dropna().drop_duplicates()
    system_groups = component_df.groupby('SYSTEM_CLEAN')['component_full'].apply(list).to_dict()


    # =========================
    # Step2: embedding + system-aware matching
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

    all_components = list(component_df['component_full'].unique())
    all_emb = encode_gpu(all_components)

    global_index = faiss.IndexFlatIP(all_emb.shape[1])
    global_index.add(all_emb)

    system_index = {}
    system_components = {}

    for sys, comps in system_groups.items():
        if len(comps) == 0:
            continue

        emb = encode_gpu(comps)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        system_index[sys] = index
        system_components[sys] = comps

    df_event['pred_system'] = df_event['need'].apply(classify_need_system)

    results = []

    for _, row in tqdm(df_event.iterrows(), total=len(df_event)):
        need = row['need']
        sys = row['pred_system']

        emb = model.encode([need], normalize_embeddings=True).astype('float32')

        if sys in system_index:
            score, idx = system_index[sys].search(emb, 1)
            best_comp = system_components[sys][idx[0][0]]
            sim = score[0][0]
        else:
            score, idx = global_index.search(emb, 1)
            best_comp = all_components[idx[0][0]]
            sim = score[0][0]

        results.append((need, best_comp, sim, sys))


    # =========================
    # Step3: mapping + overwrite original column
    # =========================
    mapping_df = pd.DataFrame(results, columns=[
        'need', 'best_component', 'similarity', 'pred_system'
    ])

    mask_nan = (
        mapping_df['need'].isna() |
        (mapping_df['need'].astype(str).str.lower() == 'nan')
    )

    mapping_df.loc[mask_nan, ['need','best_component','similarity']] = None

    mapping_df.to_csv("STEP5-need_mapping.csv", index=False)

    map_dict = dict(zip(mapping_df['need'], mapping_df['best_component']))

    df_event['test_need'] = df_event['need'].map(map_dict)

    df_event.to_csv("STEP5-examination_need_cleaned.csv", index=False)