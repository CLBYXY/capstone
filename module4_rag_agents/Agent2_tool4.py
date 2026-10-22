"""
Tool 4 - further-examination recommendation  (v2)

Change summary:
  Step2: w(d) = final_score(d), used directly, without levels or normalisation
  Step4: IG(e,d) = P(e|d) × log(P(e|d) / p̄(e))
         p_bar(e) = sum_d w(d) x P(e|d) / sum_d w(d)     weighted background mean
         IG(e,d) <= 0 -> clipped to 0
  Step5: global_score(e) = Σ_d w(d) × IG(e,d)
         = Σ_d final_score(d) × P(e|d) × log(P(e|d) / p̄(e))
"""

import os
import json
import math
import numpy as np
from typing import Any
from collections import defaultdict

from langchain_core.tools import tool
from neo4j import GraphDatabase


# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────

NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://<your-instance>.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

FURTHER_STATUS         = "further examination required"
TOP_PER_DISEASE        = 5
TOP_PER_DISEASE_OUTPUT = 3
TOP_GLOBAL             = 3


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
# 2. Step 1 - select "further" diseases
# ─────────────────────────────────────────────

def filter_further_diseases(candidate_diseases: list[dict]) -> list[dict]:
    further = []
    for d in candidate_diseases:
        dist = d.get("status_distribution", {})
        if not dist:
            continue
        if max(dist, key=dist.get) == FURTHER_STATUS:
            further.append(d)
    print(f"[Step1] further 疾病: {[d['disease'] for d in further]}")
    return further


# ─────────────────────────────────────────────
# 3. Step 3 — P(exam_type | disease)
# ─────────────────────────────────────────────

def get_exam_type_probs(disease_name: str) -> dict[str, float]:
    """P(e|d) = cnt(e,d) / cnt(*,d), based on all ExaminationEvents."""
    rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_EXAM_EVENT]->(ee:ExaminationEvent)
              -[:HAS_TYPE]->(et:ExaminationType)
        WHERE d.name = $disease
        RETURN et.name AS exam_type, COUNT(ee) AS cnt
        ORDER BY cnt DESC
    """, {"disease": disease_name})
    if not rows:
        return {}
    total = sum(r["cnt"] for r in rows)
    return {r["exam_type"]: round(r["cnt"] / total, 6) for r in rows}


# ─────────────────────────────────────────────
# 4. Core function
# ─────────────────────────────────────────────

def recommend_exams(
    candidate_diseases: list[dict],
    done_exams:         list[str],
) -> dict[str, Any]:
    """
    Returns:
        {
          "further_diseases":            [{disease, final_score}]
          "global_recommendations":      [{exam_type, global_score, contributing_diseases}]
          "per_disease_recommendations": [{disease, final_score, exams:[{exam_type,prob,ig}]}]
        }
    """
    done_set = {e.lower().strip() for e in done_exams}

    # ── Step 1 ───────────────────────────────────────────────
    further = filter_further_diseases(candidate_diseases)
    if not further:
        print("[Tool4] 无 further 疾病。")
        return {"further_diseases": [], "global_recommendations": [],
                "per_disease_recommendations": []}

    # ── Step 2: w(d) = final_score(d) ────────────────────────
    for d in further:
        w = d.get("final_score", 0.0)
        d["weight"] = w
        print(f"[Step2] '{d['disease']}'  final_score={w:.4f}  w={w:.4f}")

    w_total = sum(d["weight"] for d in further)
    if w_total == 0:
        return {"further_diseases": [], "global_recommendations": [],
                "per_disease_recommendations": []}

    # -- Step 3: query P(e|d) --------------------------------------
    # exam_prob_table[exam][disease] = P(e|d)
    # keep only the top-5 examinations of each disease for the global scoring
    exam_prob_table: dict[str, dict[str, float]] = defaultdict(dict)
    per_disease_eligible: dict[str, dict[str, float]] = {}

    for d in further:
        disease    = d["disease"]
        type_probs = get_exam_type_probs(disease)
        print(f"[Step3] '{disease}'  exam_types={len(type_probs)}")

        eligible = {
            et: p for et, p in type_probs.items()
            if et.lower().strip() not in done_set
        }
        per_disease_eligible[disease] = eligible

        top5 = sorted(eligible.items(), key=lambda x: x[1], reverse=True)[:TOP_PER_DISEASE]
        for et, p in top5:
            exam_prob_table[et][disease] = p

    # -- Step 4: compute p_bar(e) and IG(e, d) -----------------------
    #
    # p̄(e) = Σ_d w(d)×P(e|d) / Σ_d w(d)
    # IG(e,d) = P(e|d) x log( P(e|d) / p_bar(e) ), clipped at 0
    #
    all_exams = list(exam_prob_table.keys())
    p_bar: dict[str, float] = {}
    for et in all_exams:
        numerator = sum(
            d["weight"] * exam_prob_table[et].get(d["disease"], 0.0)
            for d in further
        )
        p_bar[et] = numerator / w_total

    # ig_table[exam][disease] = IG(e,d)
    ig_table: dict[str, dict[str, float]] = defaultdict(dict)
    for et in all_exams:
        pb = p_bar[et]
        if pb <= 0:
            continue
        for d in further:
            disease = d["disease"]
            p_ed    = exam_prob_table[et].get(disease, 0.0)
            if p_ed <= 0:
                ig_table[et][disease] = 0.0
            else:
                ig_val = p_ed * math.log(p_ed / pb)
                ig_table[et][disease] = round(max(ig_val, 0.0), 8)

    # ── Step 5: global_score(e) = Σ_d w(d) × IG(e,d) ────────
    global_scores: dict[str, float] = {}
    contrib_map:   dict[str, dict[str, float]] = {}

    for et in all_exams:
        score    = 0.0
        per_d    = {}
        for d in further:
            disease = d["disease"]
            w       = d["weight"]
            ig_val  = ig_table[et].get(disease, 0.0)
            c       = round(w * ig_val, 8)
            score  += c
            per_d[disease] = c
        global_scores[et] = round(score, 8)
        contrib_map[et]   = per_d

    global_top3 = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_GLOBAL]
    global_recs = [
        {
            "exam_type":             et,
            "global_score":          score,
            "contributing_diseases": contrib_map[et],
        }
        for et, score in global_top3
    ]
    for r in global_recs:
        print(f"[Step5] '{r['exam_type']}'  global_score={r['global_score']:.6f}  "
              f"contrib={r['contributing_diseases']}")

    # -- per-disease top3 (sorted by P(e|d), with IG) -------------
    per_disease_recs = []
    for d in further:
        disease  = d["disease"]
        eligible = per_disease_eligible.get(disease, {})
        top3     = sorted(eligible.items(), key=lambda x: x[1], reverse=True)[:TOP_PER_DISEASE_OUTPUT]
        per_disease_recs.append({
            "disease":     disease,
            "final_score": d["final_score"],
            "exams": [
                {
                    "exam_type": et,
                    "prob":      round(p, 6),
                    "ig":        ig_table.get(et, {}).get(disease, 0.0),
                }
                for et, p in top3
            ],
        })

    return {
        "further_diseases": [
            {"disease": d["disease"], "final_score": d["final_score"]}
            for d in further
        ],
        "global_recommendations":      global_recs,
        "per_disease_recommendations": per_disease_recs,
    }


# ─────────────────────────────────────────────
# 5. Utility functions
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
# 6. LangChain tool wrapper
# ─────────────────────────────────────────────

@tool
def recommend_further_exams(
    tool3_json:      str,
    done_exams_json: str = "[]",
) -> str:
    """
    Recommend the next examination types based on the Tool3 combined diagnosis.
    Select the "further" diseases, use final_score as weights,
    Compute global scores with the information gain IG(e,d) and output the global top3 and per-disease top3.
    """
    try:
        t3 = json.loads(tool3_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "tool3_json 解析失败"}, ensure_ascii=False)
    try:
        done_exams = json.loads(done_exams_json)
    except json.JSONDecodeError:
        done_exams = []
    candidate_diseases = t3.get("candidate_diseases", [])
    if not candidate_diseases:
        return json.dumps({"error": "candidate_diseases 为空"}, ensure_ascii=False)
    result = recommend_exams(candidate_diseases, done_exams)
    return json.dumps(convert_numpy(result), ensure_ascii=False, indent=2)

