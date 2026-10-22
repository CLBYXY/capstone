"""
Tool 1 - symptom matching  (v2)

Change summary (3 changes):
  1. Offline stage: compute_sf_idf()
       SF(s,d)  = cnt(s,d) / cnt(*,d)          frequency of the symptom under the disease
       IDF(s)   = log(N / df(s) + 1)            df(s) = number of diseases with this symptom, N = total number of diseases
       raw_score = SF(s,d) × IDF(s)
       normalised to [0,1] within each symptom node across diseases, keeping the same scale as final_sim
       stored in the FAISS metadata key "sf_idf_scores" (replaces the former "disease_probs")

  2. Runtime: score_diseases()
       contribution(d) = 0.5 × final_sim + 0.5 × SF-IDF(s,d)
       (formula structure unchanged; the second term changes from P(d|s) to the normalised SF-IDF)

  3. Time similarity: _time_similarity()
       changed from manual step rules to exponential decay:
       sim = exp(−λ × Δt_days)
       lambda by the unit of the input duration:
         day   → λ = 0.10
         week  → λ = 0.015
         month → λ = 0.005
         year  → λ = 0.0014
       still returns the neutral value 0.5 when the time is missing

final_similarity  = 0.5 × syp_sim + 0.5 × best_time_sim
contribution(d)   = 0.5 × final_sim + 0.5 × SF-IDF(s,d)
disease_score     = SUM of top5 contributions  ∈ [0, 5] + Softmax

status_distribution per disease:
  At runtime, accumulate the status counts of all matched symptoms under the disease and normalise them together
  P(status | d, matched_symps) = Σcnt(symp_i,d,status) / Σ Σcnt(symp_i,d,*)
"""

import os
import json
import math
import re
import numpy as np
from typing import Any
from collections import defaultdict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase


# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────

NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://<your-instance>.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

SYP_SIM_THRESHOLD = 0.6
TOP_K             = 10

# exponential decay lambda, four levels by the unit of the input duration
LAMBDA = {
    "day":   0.10,
    "week":  0.015,
    "month": 0.005,
    "year":  0.0014,
}


# ─────────────────────────────────────────────
# 1. Neo4j connection
# ─────────────────────────────────────────────

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query(self, cypher: str, params: dict = {}) -> list[dict]:
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, params)]

    def close(self):
        self.driver.close()


neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


# ─────────────────────────────────────────────
# 2. Offline SF-IDF computation (replaces the former P(disease|symp))
# ─────────────────────────────────────────────

def compute_sf_idf() -> dict[str, dict[str, float]]:
    """
    Offline SF-IDF computation; returns scores normalised to [0,1] within each symptom node.

    SF(s, d) = cnt(s, d) / cnt(*, d)
        number of SymptomEvents of symptom s under disease d / total number of SymptomEvents of disease d
        -> reflects the relative frequency of the symptom under the disease

    IDF(s) = log(N / (df(s) + 1))
        N    = total number of diseases
        df(s)= number of diseases in which symptom s appears
        -> common symptoms get low weights, rare symptoms get high weights

    raw_score(s, d) = SF(s, d) × IDF(s)

    Normalisation: for each symptom node s, normalise raw_score over all its diseases to [0,1]
        normalized(s, d) = raw_score(s, d) / max_{d'} raw_score(s, d')
        keeps the same scale as final_sim in [0,1]

    Returns:
        {symp_id: {disease_name: normalized_sf_idf}}
    """
    # 1. Query all (symp_id, disease, co_count)
    rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
              -[:HAS_SYMPTOM]->(s:Symptom)
        RETURN elementId(s) AS symp_id,
               d.name        AS disease_name,
               COUNT(se)     AS co_count
    """)

    # 2. Query the total number of SymptomEvents of each disease (denominator cnt(*,d))
    disease_total_rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
        RETURN d.name AS disease_name, COUNT(se) AS total_count
    """)
    disease_total: dict[str, int] = {r["disease_name"]: r["total_count"] for r in disease_total_rows}

    # 3. Accumulate cnt(s, d)
    co_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        co_counts[r["symp_id"]][r["disease_name"]] += r["co_count"]

    # 4. IDF(s): df(s) = number of diseases with this symptom, N = total number of diseases
    N = len(disease_total)
    # guard: symp_id may be an int (elementId returns int in some Neo4j versions); convert to str
    co_counts_str: dict[str, dict[str, int]] = {
        str(sid): {d: cnt for d, cnt in d_map.items()}
        for sid, d_map in co_counts.items()
        if isinstance(d_map, dict)   # skip non-dict values (guard against a defaultdict being treated as int)
    }
    df_map: dict[str, int] = {
        symp_id: len(d_map) for symp_id, d_map in co_counts_str.items()
    }

    # 5. Compute raw SF-IDF, then normalise within each symptom node
    result: dict[str, dict[str, float]] = {}
    for symp_id, d_map in co_counts_str.items():
        df   = df_map.get(symp_id, 1)
        idf  = math.log(max(N / (df + 1), 1e-9))  # prevent a negative log (clip when N < df+1)
        idf  = max(idf, 0.0)                        # Negative IDF is meaningless; clip to 0

        raw: dict[str, float] = {}
        for disease, cnt in d_map.items():
            d_total = max(int(disease_total.get(disease, 1)), 1)
            sf      = cnt / d_total
            raw[disease] = sf * idf

        if not raw:
            continue

        # normalise to [0,1]: take the maximum of the positive values
        max_val = max((v for v in raw.values() if v > 0), default=0.0)
        if max_val <= 0:
            # all values are 0 (when IDF = 0): assign a uniform value
            result[symp_id] = {d: round(1.0 / len(raw), 6) for d in raw}
        else:
            result[symp_id] = {
                d: round(max(v, 0.0) / max_val, 6) for d, v in raw.items()
            }

    return result


# ─────────────────────────────────────────────
# 3. Build / load the vector index
# ─────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_symptom_vector_index(save_path: str = "symptom_faiss_index") -> FAISS:
    """
    Offline step; only needs to run once (or again after the KG is updated).
    metadata key changed from "disease_probs" to "sf_idf_scores".
    """
    rows = neo4j_client.query(
        "MATCH (s:Symptom) RETURN s.name AS name, elementId(s) AS neo4j_id"
    )
    if not rows:
        raise ValueError("Neo4j 中没有找到 Symptom 节点。")

    sf_idf = compute_sf_idf()
    texts     = [r["name"] for r in rows]
    metadatas = [
        {
            "neo4j_id":       r["neo4j_id"],
            "name":           r["name"],
            "sf_idf_scores":  json.dumps(sf_idf.get(r["neo4j_id"], {})),
        }
        for r in rows
    ]

    vector_store = FAISS.from_texts(texts, _get_embeddings(), metadatas=metadatas)
    vector_store.save_local(save_path)
    print(f"[build] {len(texts)} 个症状节点，SF-IDF 索引已保存至 {save_path}/")
    return vector_store


def load_symptom_vector_index(save_path: str = "symptom_faiss_index") -> FAISS:
    return FAISS.load_local(
        save_path, _get_embeddings(), allow_dangerous_deserialization=True
    )


# ─────────────────────────────────────────────
# 4. Step 1 - LLM extracts symptom + duration pairs
# ─────────────────────────────────────────────

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a medical symptom extraction expert.
Extract all independent symptoms and their durations from the input.

Rules:
1. symptom: short core clinical term, no numbers/dates.
2. Split multiple symptoms in one sentence into separate items.
3. For duration:
   - concrete value+unit → {{value: number, unit: day|week|month|year}}
   - fuzzy word (e.g. ongoing, sometimes, recently, occasional, chronic,
     last night, today) → keep original in duration_raw, set value/unit null
   - no duration → all null
4. Output ONLY valid JSON, no markdown.

Input: {user_input}

Output:
{{
  "symptoms": [
    {{
      "core": "<symptom>",
      "original": "<original text>",
      "duration": {{
        "value": <number|null>,
        "unit": "<day|week|month|year|null>",
        "duration_raw": "<original time phrase|null>"
      }}
    }}
  ]
}}
"""
)


def extract_symptoms(user_input: str, llm: ChatOpenAI) -> list[dict]:
    raw = (EXTRACT_PROMPT | llm).invoke({"user_input": user_input}).content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        symptoms = json.loads(raw).get("symptoms", [])
        for s in symptoms:
            s.setdefault("duration", {"value": None, "unit": None, "duration_raw": None})
        return symptoms
    except json.JSONDecodeError:
        return [{"core": user_input, "original": user_input,
                 "duration": {"value": None, "unit": None, "duration_raw": None}}]


# ─────────────────────────────────────────────
# 5. Time similarity (exponential decay, replaces the former step rules)
# ─────────────────────────────────────────────

UNIT_TO_DAYS  = {"day": 1, "week": 7, "month": 30, "year": 365}
UNIT_PRIORITY = {"day": 1, "week": 2, "month": 3,  "year": 4}


def _time_diff_in_days(iv, iu, kv, ku) -> float:
    if any(x is None for x in [iv, iu, kv, ku]):
        return float("inf")
    iu, ku = str(iu).lower(), str(ku).lower()
    if iu not in UNIT_TO_DAYS or ku not in UNIT_TO_DAYS:
        return float("inf")
    return abs(float(iv) * UNIT_TO_DAYS[iu] - float(kv) * UNIT_TO_DAYS[ku])


def _time_similarity(iv, iu, kv, ku) -> float:
    """
    Exponential-decay time similarity: sim = exp(-lambda x delta_t_days)
    lambda has four levels by the unit of the input duration (day/week/month/year).
    returns the neutral value 0.5 if any value is missing.
    """
    if any(x is None for x in [iv, iu, kv, ku]):
        return 0.5
    iu, ku = str(iu).lower(), str(ku).lower()
    if iu not in UNIT_TO_DAYS or ku not in UNIT_TO_DAYS:
        return 0.5

    delta_days = abs(float(iv) * UNIT_TO_DAYS[iu] - float(kv) * UNIT_TO_DAYS[ku])
    lam        = LAMBDA.get(iu, LAMBDA["month"])   # choose lambda by the input unit
    return round(math.exp(-lam * delta_days), 6)


def _parse_time_str(time_str: str | None) -> tuple[float | None, str | None]:
    if time_str is None:
        return None, None
    s = str(time_str).strip().lower()
    UNIT_ALIASES = {
        "day": "day", "days": "day",
        "week": "week", "weeks": "week",
        "month": "month", "months": "month",
        "year": "year", "years": "year",
    }
    parts = s.split()
    if len(parts) == 2:
        try:
            val  = float(parts[0])
            unit = UNIT_ALIASES.get(parts[1])
            if unit:
                return val, unit
        except ValueError:
            pass
    return None, None


# ─────────────────────────────────────────────
# 6. KG time query
# ─────────────────────────────────────────────

def get_kg_best_time_sim(
    neo4j_id:     str,
    input_value:  float | None,
    input_unit:   str   | None,
    duration_raw: str   | None,
    embeddings:   HuggingFaceEmbeddings,
) -> tuple[float, str | None]:
    """
    Strategy A (concrete duration): find the KG duration with the smallest difference -> exponential-decay similarity
    Strategy B (fuzzy duration): SapBERT embedding matching
    returns the neutral value 0.5 when there is no time information
    """
    rows = neo4j_client.query("""
        MATCH (se:SymptomEvent)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE elementId(s) = $id
        OPTIONAL MATCH (se)-[:HAS_TIME]->(t:Time)
        RETURN t.value AS v
    """, {"id": neo4j_id})

    kg_times = [r.get("v") for r in rows if r.get("v") is not None]
    if not kg_times:
        return 0.5, None

    # Strategy A: concrete duration -> find the KG duration with the smallest difference, then apply exponential decay
    if input_value is not None and input_unit is not None:
        best_diff, best_raw, best_kv, best_ku = float("inf"), None, None, None
        for raw in kg_times:
            kv, ku = _parse_time_str(raw)
            diff   = _time_diff_in_days(input_value, input_unit, kv, ku)
            if diff < best_diff:
                best_diff, best_raw, best_kv, best_ku = diff, raw, kv, ku
        return _time_similarity(input_value, input_unit, best_kv, best_ku), best_raw

    # Strategy B: fuzzy duration -> SapBERT embedding
    if duration_raw is not None:
        from sentence_transformers import util as st_util
        model     = embeddings.client
        input_emb = model.encode(duration_raw, normalize_embeddings=True)
        kg_embs   = model.encode(kg_times,     normalize_embeddings=True)
        sims      = st_util.cos_sim(input_emb, kg_embs)[0].tolist()
        best_idx  = int(np.argmax(sims))
        return float(sims[best_idx]), kg_times[best_idx]

    return 0.5, None


# ─────────────────────────────────────────────
# 7. KG status count query (runtime)
# ─────────────────────────────────────────────

def get_status_counts(neo4j_id: str, disease_name: str) -> dict[str, int]:
    rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
              -[:HAS_SYMPTOM]->(s:Symptom)
        WHERE elementId(s) = $symp_id
          AND d.name = $disease
        MATCH (se)-[:BELONGS_TO]->(c:Conversation)-[:IN_STATUS]->(st:ConfirmedStatus)
        RETURN st.value AS status, COUNT(c) AS cnt
    """, {"symp_id": neo4j_id, "disease": disease_name})
    return {r["status"]: r["cnt"] for r in rows if r["status"] is not None}


# ─────────────────────────────────────────────
# 8. Step 2 - vector retrieval + time matching + status counts
# ─────────────────────────────────────────────

def vector_search_symptom(
    symptom_core:   str,
    input_duration: dict,
    vector_store:   FAISS,
    embeddings:     HuggingFaceEmbeddings,
) -> list[dict]:
    """
    For each FAISS candidate:
      1. filter by syp_sim (> SYP_SIM_THRESHOLD)
      2. best_time_sim (exponential decay or embedding)
      3. final_sim = 0.5 × syp_sim + 0.5 × best_time_sim
      4. read sf_idf_scores (replaces the former disease_probs)
      5. query status_counts per (symp_node, disease)
    """
    results    = vector_store.similarity_search_with_relevance_scores(symptom_core, k=TOP_K)
    candidates = []

    input_value  = input_duration.get("value")
    input_unit   = input_duration.get("unit")
    duration_raw = input_duration.get("duration_raw")

    for doc, score in results:
        syp_sim = float(score)
        if syp_sim <= SYP_SIM_THRESHOLD:
            continue

        neo4j_id = doc.metadata["neo4j_id"]
        best_time_sim, best_kg_time = get_kg_best_time_sim(
            neo4j_id, input_value, input_unit, duration_raw, embeddings
        )
        final_sim = 0.5 * syp_sim + 0.5 * best_time_sim

        # read SF-IDF scores (compatible with old indexes: fall back to disease_probs if present)
        raw_sf = doc.metadata.get("sf_idf_scores") or doc.metadata.get("disease_probs", "{}")
        sf_idf_scores = json.loads(raw_sf) if isinstance(raw_sf, str) else raw_sf

        # query the status counts under each disease
        status_counts_per_disease: dict[str, dict[str, int]] = {}
        for disease in sf_idf_scores:
            counts = get_status_counts(neo4j_id, disease)
            if counts:
                status_counts_per_disease[disease] = counts

        candidates.append({
            "kg_name":                   doc.metadata["name"],
            "neo4j_id":                  neo4j_id,
            "syp_sim":                   round(syp_sim,       4),
            "best_time_sim":             round(best_time_sim, 4),
            "best_kg_time":              best_kg_time,
            "final_sim":                 round(final_sim,     4),
            "sf_idf_scores":             sf_idf_scores,
            "status_counts_per_disease": status_counts_per_disease,
        })

    return candidates


# ─────────────────────────────────────────────
# 9. Step 3 - disease scoring + status_distribution
# ─────────────────────────────────────────────

def _softmax(values: list[float]) -> list[float]:
    arr     = np.array(values, dtype=np.float64)
    exp_arr = np.exp(arr - arr.max())
    return [round(float(p), 6) for p in exp_arr / exp_arr.sum()]


def score_diseases(
    symptom_candidates: list[dict],
    min_score:          float = 0.2,
) -> list[dict]:
    """
    disease score:
      contribution(d) = 0.5 × final_sim + 0.5 × SF-IDF(s, d)
      disease_score   = SUM of top5 contributions

    status_distribution per disease:
      accumulate the raw status counts of all matched symptoms under the disease and normalise them together
    """
    disease_contributions: dict[str, list[float]]    = defaultdict(list)
    disease_symp_map:      dict[str, set]            = defaultdict(set)
    disease_status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cand in symptom_candidates:
        final_sim             = cand["final_sim"]
        sf_idf_scores         = cand["sf_idf_scores"]
        kg_name               = cand["kg_name"]
        status_counts_per_dis = cand.get("status_counts_per_disease", {})

        for disease, sfidf in sf_idf_scores.items():
            contrib = 0.5 * final_sim + 0.5 * sfidf
            disease_contributions[disease].append(contrib)
            disease_symp_map[disease].add(kg_name)

            for status, cnt in status_counts_per_dis.get(disease, {}).items():
                disease_status_counts[disease][status] += cnt

    results = []
    for disease, contribs in disease_contributions.items():
        top5 = round(sum(sorted(contribs, reverse=True)[:5]), 4)
        if top5 < min_score:
            continue

        raw_counts = disease_status_counts[disease]
        total_cnt  = sum(raw_counts.values())
        status_dist = (
            {st: round(cnt / total_cnt, 6) for st, cnt in raw_counts.items()}
            if total_cnt > 0 else {}
        )

        results.append({
            "disease":             disease,
            "total_score":         top5,
            "matched_symptoms":    list(disease_symp_map[disease]),
            "status_distribution": status_dist,
        })

    if not results:
        return []

    results.sort(key=lambda x: x["total_score"], reverse=True)
    results = results[:10]

    probs = _softmax([d["total_score"] for d in results])
    for i, d in enumerate(results):
        d["softmax_prob"] = probs[i]

    return results


# ─────────────────────────────────────────────
# 10. Full pipeline
# ─────────────────────────────────────────────

def symptom_matching_pipeline(
    user_input:   str,
    llm:          ChatOpenAI,
    vector_store: FAISS,
    embeddings:   HuggingFaceEmbeddings,
) -> dict[str, Any]:

    # Step 1: LLM extracts symptoms
    extracted = extract_symptoms(user_input, llm)
    for s in extracted:
        print(f"[Step1] '{s['core']}'  duration={s['duration']}")

    # Step 2: vector retrieval + time matching
    all_candidates = []
    detail         = []

    for sym in extracted:
        core      = sym["core"]
        duration  = sym.get("duration", {"value": None, "unit": None, "duration_raw": None})
        candidates = vector_search_symptom(core, duration, vector_store, embeddings)

        for c in candidates:
            strategy = "diff" if duration.get("value") is not None else "embed"
            print(
                f"[Step2|{strategy}] '{core}' → '{c['kg_name']}'  "
                f"syp={c['syp_sim']}  best_time={c['best_time_sim']}  "
                f"kg_time='{c['best_kg_time']}'  final={c['final_sim']}"
            )

        all_candidates.extend(candidates)
        detail.append({
            "input_symptom":  core,
            "original":       sym["original"],
            "input_duration": duration,
            "candidates":     candidates,
        })

    # Step 3: disease scoring
    candidate_diseases = score_diseases(all_candidates)
    for d in candidate_diseases:
        print(
            f"[Step3] {d['disease']}  score={d['total_score']}  "
            f"p={d['softmax_prob']}  status={d['status_distribution']}"
        )

    return {
        "extracted_symptoms": extracted,
        "symptom_candidates": detail,
        "candidate_diseases": candidate_diseases,
    }


# ─────────────────────────────────────────────
# 11. numpy conversion utility
# ─────────────────────────────────────────────

def convert_numpy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# ─────────────────────────────────────────────
# 12. LangChain tool wrapper
# ─────────────────────────────────────────────

_vector_store: FAISS | None                = None
_embeddings:   HuggingFaceEmbeddings | None = None
_llm:          ChatOpenAI | None            = None


def init_tool(openai_api_key: str, index_path: str = "symptom_faiss_index"):
    global _vector_store, _embeddings, _llm
    _embeddings   = _get_embeddings()
    _vector_store = load_symptom_vector_index(index_path)
    _llm          = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
    print("[init] 完成。")


@tool
def retrieve_by_symptoms(user_symptom_description: str) -> str:
    """
    Retrieve candidate diseases from the knowledge graph based on the natural-language symptom description of the user (including durations).
    Each candidate disease contains status_distribution,
    i.e. the status probability distribution obtained by normalising the historical status counts of all matched symptoms under the disease.
    """
    if _vector_store is None or _llm is None:
        return json.dumps({"error": "请先调用 init_tool()"}, ensure_ascii=False)

    result = symptom_matching_pipeline(
        user_symptom_description, _llm, _vector_store, _embeddings
    )

    detail_map: dict[str, list[dict]] = {
        d["input_symptom"]: d["candidates"]
        for d in result["symptom_candidates"]
    }

    output = {
        "extracted_symptoms": [
            {
                "core":     s["core"],
                "duration": s["duration"],
                "retrieved_kg_entry": (
                    {
                        "kg_symptom": detail_map[s["core"]][0]["kg_name"],
                        "kg_time":    detail_map[s["core"]][0]["best_kg_time"],
                    }
                    if detail_map.get(s["core"]) else None
                ),
            }
            for s in result["extracted_symptoms"]
        ],
        "candidate_diseases": [
            {
                "disease":             d["disease"],
                "total_score":         d["total_score"],
                "softmax_prob":        d["softmax_prob"],
                "status_distribution": d["status_distribution"],
                "matched_symptoms":    d.get("matched_symptoms", []),
            }
            for d in result["candidate_diseases"]
        ],
    }
    return json.dumps(convert_numpy(output), ensure_ascii=False, indent=2)