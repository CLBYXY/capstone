"""
Tool 2 - examination result matching (with status_distribution per disease)
Pipeline:
  Step1: LLM extracts (examination_type, examination_result) pairs
  Step2: SapBERT matching of ExaminationType (kept only if type_sim >= 0.8)
  Step3: find all ExaminationResults of the matched type and compute result_sim with SapBERT
         also query the raw status counts per (type_node, disease)
  Step4: disease scoring + status_distribution
         exam_score      = 0.5 * type_sim + 0.5 * result_sim
         disease_score   = SUM of top5  ∈ [0, 5] + Softmax
         status_dist(d)  = Σcnt(type_i, d, status) / Σ Σcnt(type_i, d, *)

KG path (disease):
  Disease -[:HAS_EXAM_EVENT]-> ExaminationEvent
          -[:HAS_TYPE]->       ExaminationType
          -[:HAS_RESULT]->     ExaminationResult

KG path (status counts):
  ExaminationEvent -[:BELONGS_TO]-> Conversation -[:IN_STATUS]-> ConfirmedStatus
"""

import os
import json
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

TYPE_SIM_THRESHOLD = 0.8
TOP_K              = 10


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
# 2. Build / load the vector index
#    metadata only stores neo4j_id and name; status is computed at runtime
# ─────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_exam_vector_index(save_path: str = "exam_faiss_index") -> FAISS:
    """
    Offline step; only needs to run once.
    metadata only stores neo4j_id and name; status_probs is no longer stored.
    """
    rows = neo4j_client.query(
        "MATCH (et:ExaminationType) RETURN et.name AS name, elementId(et) AS neo4j_id"
    )
    if not rows:
        raise ValueError("Neo4j中没有找到ExaminationType节点。")

    texts     = [r["name"] for r in rows]
    metadatas = [{"neo4j_id": r["neo4j_id"], "name": r["name"]} for r in rows]

    vector_store = FAISS.from_texts(texts, _get_embeddings(), metadatas=metadatas)
    vector_store.save_local(save_path)
    print(f"[build] {len(texts)} 个ExaminationType节点，索引已保存至 {save_path}/")
    return vector_store


def load_exam_vector_index(save_path: str = "exam_faiss_index") -> FAISS:
    return FAISS.load_local(
        save_path, _get_embeddings(), allow_dangerous_deserialization=True
    )


# ─────────────────────────────────────────────
# 3. Step 1 - LLM extracts examination type + result
# ─────────────────────────────────────────────

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a medical examination information extraction expert.
Extract all examination types and their corresponding results from the following medical description.

Rules:
1. examination_type: the name of the test or examination (e.g. "fasting blood glucose", "HbA1c", "urine protein").
2. examination_result: the finding or value (e.g. "elevated", "7.2 mmol/L", "positive", "normal").
   - If no result is mentioned for a type, set result to null.
3. Split multiple examinations into separate items.
4. Output ONLY valid JSON, no markdown.

Input: {user_input}

Output:
{{
  "examinations": [
    {{
      "type": "<examination type>",
      "result": "<examination result or null>",
      "original": "<original fragment>"
    }}
  ]
}}
"""
)


def extract_examinations(user_input: str, llm: ChatOpenAI) -> list[dict]:
    raw = (EXTRACT_PROMPT | llm).invoke({"user_input": user_input}).content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        exams = json.loads(raw).get("examinations", [])
        for e in exams:
            e.setdefault("result", None)
        return exams
    except json.JSONDecodeError:
        return [{"type": user_input, "result": None, "original": user_input}]


# ─────────────────────────────────────────────
# 4. Step 2 - ExaminationType vector matching
# ─────────────────────────────────────────────

def match_exam_type(
    exam_type_input:   str,
    type_vector_store: FAISS,
) -> list[dict]:
    """Filter out type_sim < 0.8 and return candidate type nodes (without status information)."""
    results    = type_vector_store.similarity_search_with_relevance_scores(
        exam_type_input, k=TOP_K
    )
    candidates = []
    for doc, score in results:
        type_sim = float(score)
        if type_sim < TYPE_SIM_THRESHOLD:
            continue
        candidates.append({
            "kg_type_name": doc.metadata["name"],
            "neo4j_id":     doc.metadata["neo4j_id"],
            "type_sim":     round(type_sim, 4),
        })
    return candidates


# ─────────────────────────────────────────────
# 5. Step 3a - query KG ExaminationResult + Disease
# ─────────────────────────────────────────────

def get_results_for_type(type_neo4j_id: str) -> list[dict]:
    """From an ExaminationType, find all linked ExaminationResults and Diseases."""
    rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_EXAM_EVENT]->(ee:ExaminationEvent)
              -[:HAS_TYPE]->(et:ExaminationType)
        WHERE elementId(et) = $type_id
        OPTIONAL MATCH (ee)-[:HAS_RESULT]->(er:ExaminationResult)
        RETURN d.name        AS disease_name,
               er.value      AS result_name,
               elementId(er) AS result_neo4j_id
    """, {"type_id": type_neo4j_id})
    return rows


# ─────────────────────────────────────────────
# 6. Step 3b - ExaminationResult vector matching
# ─────────────────────────────────────────────

def match_exam_result(
    result_input: str | None,
    kg_results:   list[dict],
    embeddings:   HuggingFaceEmbeddings,
) -> list[dict]:
    """
    Match the user result with all results of this type in the KG by vector similarity.
    result_input is set to the neutral value 0.5 when it is None.
    """
    if not kg_results:
        return []

    valid = [r for r in kg_results if r.get("result_name")]
    if not valid:
        return [dict(r, result_sim=0.5) for r in kg_results]

    if result_input is None:
        return [dict(r, result_sim=0.5) for r in valid]

    from sentence_transformers import util as st_util
    model     = embeddings.client
    input_emb = model.encode(result_input, normalize_embeddings=True)
    kg_texts  = [r["result_name"] for r in valid]
    kg_embs   = model.encode(kg_texts,    normalize_embeddings=True)
    sims      = st_util.cos_sim(input_emb, kg_embs)[0].tolist()

    return [
        dict(r, result_sim=round(float(sims[i]), 4))
        for i, r in enumerate(valid)
    ]


# ─────────────────────────────────────────────
# 7. Step 3c - status count query (runtime)
# ─────────────────────────────────────────────

def get_status_counts_by_exam(type_neo4j_id: str, disease_name: str) -> dict[str, int]:
    """
    Query the raw count of each status for a (type_node, disease) combination.
    Path: Disease -> ExaminationEvent -> ExaminationType
          ExaminationEvent -[:BELONGS_TO]-> Conversation -[:IN_STATUS]-> ConfirmedStatus
    """
    rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_EXAM_EVENT]->(ee:ExaminationEvent)
              -[:HAS_TYPE]->(et:ExaminationType)
        WHERE elementId(et) = $type_id
          AND d.name = $disease
        MATCH (ee)-[:BELONGS_TO]->(c:Conversation)-[:IN_STATUS]->(st:ConfirmedStatus)
        RETURN st.value AS status, COUNT(ee) AS cnt
    """, {"type_id": type_neo4j_id, "disease": disease_name})

    return {r["status"]: r["cnt"] for r in rows if r["status"] is not None}


# ─────────────────────────────────────────────
# 8. Step 4 - disease scoring + status_distribution
# ─────────────────────────────────────────────

def _softmax(values: list[float]) -> list[float]:
    arr     = np.array(values, dtype=np.float64)
    exp_arr = np.exp(arr - arr.max())
    return [round(float(p), 6) for p in exp_arr / exp_arr.sum()]


def score_diseases(
    exam_candidates: list[dict],
    min_score:       float = 0.2,
) -> list[dict]:
    """
    disease score:
      exam_score(d)  = 0.5 * type_sim + 0.5 * result_sim
      disease_score  = SUM of top5 exam_scores

    status_distribution per disease:
      Σ cnt(type_i, d, status) / Σ Σ cnt(type_i, d, *)
      across all matched type candidates for that disease
    """
    disease_contributions: dict[str, list[float]]    = defaultdict(list)
    disease_exam_map:      dict[str, set]            = defaultdict(set)
    # {disease: {status: accumulated_count}}
    disease_status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cand in exam_candidates:
        disease    = cand["disease_name"]
        exam_score = 0.5 * cand["type_sim"] + 0.5 * cand["result_sim"]

        disease_contributions[disease].append(exam_score)
        disease_exam_map[disease].add(
            f"{cand['kg_type_name']}:{cand.get('result_name', 'N/A')}"
        )

        # accumulate the status counts
        for status, cnt in cand.get("status_counts", {}).items():
            disease_status_counts[disease][status] += cnt

    results = []
    for disease, scores in disease_contributions.items():
        top5 = round(sum(sorted(scores, reverse=True)[:5]), 4)
        if top5 < min_score:
            continue

        # normalise the status counts
        raw_counts = disease_status_counts[disease]
        total_cnt  = sum(raw_counts.values())
        status_dist = (
            {st: round(cnt / total_cnt, 6) for st, cnt in raw_counts.items()}
            if total_cnt > 0 else {}
        )

        results.append({
            "disease":             disease,
            "total_score":         top5,
            "matched_exams":       list(disease_exam_map[disease]),
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
# 9. Full pipeline
# ─────────────────────────────────────────────

def exam_matching_pipeline(
    user_input:        str,
    llm:               ChatOpenAI,
    type_vector_store: FAISS,
    embeddings:        HuggingFaceEmbeddings,
) -> dict[str, Any]:

    # Step 1
    extracted = extract_examinations(user_input, llm)
    for e in extracted:
        print(f"[Step1] type='{e['type']}'  result='{e['result']}'")

    # Step 2 + Step 3
    all_candidates = []
    detail         = []

    for exam in extracted:
        type_input   = exam["type"]
        result_input = exam["result"]

        type_matches = match_exam_type(type_input, type_vector_store)

        if not type_matches:
            print(f"[Step2] '{type_input}' → 无type_sim>=0.8候选，跳过")
            detail.append({
                "input_type":   type_input,
                "input_result": result_input,
                "original":     exam["original"],
                "candidates":   [],
            })
            continue

        exam_candidates = []
        for tm in type_matches:
            print(f"[Step2] '{type_input}' → '{tm['kg_type_name']}'  type_sim={tm['type_sim']}")

            # Step 3a: query the result pool
            kg_results      = get_results_for_type(tm["neo4j_id"])
            matched_results = match_exam_result(result_input, kg_results, embeddings)

            # get all diseases linked to this type
            diseases_for_type = list({r["disease_name"] for r in kg_results if r.get("disease_name")})

            for mr in matched_results:
                disease = mr["disease_name"]
                print(
                    f"  [Step3] result='{result_input}' → kg='{mr.get('result_name')}'  "
                    f"result_sim={mr['result_sim']}  disease={disease}"
                )

                # Step 3c: query status counts
                status_counts = get_status_counts_by_exam(tm["neo4j_id"], disease)

                exam_candidates.append({
                    "kg_type_name": tm["kg_type_name"],
                    "neo4j_id":     tm["neo4j_id"],
                    "type_sim":     tm["type_sim"],
                    "result_name":  mr.get("result_name"),
                    "result_sim":   mr["result_sim"],
                    "disease_name": disease,
                    "status_counts": status_counts,
                })

        all_candidates.extend(exam_candidates)
        detail.append({
            "input_type":   type_input,
            "input_result": result_input,
            "original":     exam["original"],
            "candidates":   exam_candidates,
        })

    # Step 4
    candidate_diseases = score_diseases(all_candidates)
    for d in candidate_diseases:
        print(
            f"[Step4] {d['disease']}  score={d['total_score']}  "
            f"p={d['softmax_prob']}  status={d['status_distribution']}"
        )

    return {
        "extracted_examinations": extracted,
        "exam_candidates":        detail,
        "candidate_diseases":     candidate_diseases,
    }


# ─────────────────────────────────────────────
# 10. numpy conversion utility
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
# 11. LangChain tool wrapper
# ─────────────────────────────────────────────

_type_vector_store: FAISS | None              = None
_embeddings:        HuggingFaceEmbeddings | None = None
_llm:               ChatOpenAI | None         = None


def init_tool(openai_api_key: str, index_path: str = "exam_faiss_index"):
    global _type_vector_store, _embeddings, _llm
    _embeddings        = _get_embeddings()
    _type_vector_store = load_exam_vector_index(index_path)
    _llm               = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
    print("[init] 完成。")


@tool
def retrieve_by_examinations(user_exam_description: str) -> str:
    """
    Retrieve candidate diseases from the knowledge graph based on the examination types and results described by the user.
    Examination types with type_sim < 0.8 are excluded.
    exam_score = 0.5*type_sim + 0.5*result_sim, maximum 5 points.
    Each candidate disease contains status_distribution,
    obtained by normalising the historical ExaminationEvent status counts of all matched examinations under the disease.
    """
    if _type_vector_store is None or _llm is None:
        return json.dumps({"error": "请先调用 init_tool()"}, ensure_ascii=False)

    result = exam_matching_pipeline(
        user_exam_description, _llm, _type_vector_store, _embeddings
    )

    # map input_type -> candidates for display in retrieve
    detail_map: dict[str, list[dict]] = {
        d["input_type"]: d["candidates"]
        for d in result["exam_candidates"]
    }

    output = {
        "extracted_examinations": [
            {
                "type":   e["type"],
                "result": e["result"],
                "retrieved_kg_entry": (
                    {
                        "kg_type":   detail_map[e["type"]][0]["kg_type_name"],
                        "kg_result": detail_map[e["type"]][0].get("result_name"),
                    }
                    if detail_map.get(e["type"]) else None
                ),
            }
            for e in result["extracted_examinations"]
        ],
        "candidate_diseases": [
            {
                "disease":             d["disease"],
                "total_score":         d["total_score"],
                "softmax_prob":        d["softmax_prob"],
                "status_distribution": d["status_distribution"],
                "matched_exams":       d.get("matched_exams", []),
            }
            for d in result["candidate_diseases"]
        ],
    }
    return json.dumps(convert_numpy(output), ensure_ascii=False, indent=2)