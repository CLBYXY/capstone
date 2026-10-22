"""
Tool 3 - combined diagnosis
Merge the outputs of Tool1 (symptom matching) and Tool2 (examination matching).

Disease:
  final_score(d) = mean(tool1_score, tool2_score)
  missing → score = 0, mean = score / 2
  sort desc → top5 → softmax

Status (per disease, independently):
  mean_prob(status) = mean(tool1_prob, tool2_prob)
  missing → prob = 0, mean = prob / 2
  P(status|d) = mean_prob(status) / Σ mean_prob(*)  [normalize within disease]
  top5 per disease
"""

import json
import numpy as np
from typing import Any

from langchain_core.tools import tool


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def _softmax(values: list[float]) -> list[float]:
    arr     = np.array(values, dtype=np.float64)
    exp_arr = np.exp(arr - arr.max())
    return [round(float(p), 6) for p in exp_arr / exp_arr.sum()]


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
# Status merging (per disease)
# ─────────────────────────────────────────────

def _merge_status_distribution(
    t1_dist: dict[str, float],
    t2_dist: dict[str, float],
    top_n:   int = 5,
) -> dict[str, float]:
    """
    Merge the status_distribution within a single disease:
      mean_prob(status) = mean(t1_prob, t2_prob)  [missing -> 0]
      P(status|d) = mean_prob / sum(mean_prob)        [normalised]
      return the top_n statuses
    """
    all_statuses = set(t1_dist) | set(t2_dist)

    mean_probs = {
        st: (t1_dist.get(st, 0.0) + t2_dist.get(st, 0.0)) / 2.0
        for st in all_statuses
    }

    total = sum(mean_probs.values())
    if total == 0:
        return {}

    normalized = {
        st: round(p / total, 6)
        for st, p in mean_probs.items()
    }

    # top_n
    sorted_statuses = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(sorted_statuses)


# ─────────────────────────────────────────────
# Core merging logic
# ─────────────────────────────────────────────

def combine_results(
    tool1_output: dict,
    tool2_output: dict,
    top_n:        int   = 5,
    min_score:    float = 0.05,
) -> dict[str, Any]:
    """
    Merge the full outputs of Tool1 and Tool2.

    Args:
        tool1_output: parsed result of Tool1 retrieve_by_symptoms
                      must contain candidate_diseases (each with total_score, status_distribution)
        tool2_output: parsed result of Tool2 retrieve_by_examinations
                      must contain candidate_diseases (each with total_score, status_distribution)
        top_n:        number of diseases and statuses kept (top_n each)
        min_score:    diseases with final_score below this value are filtered out

    Returns:
        {
          "candidate_diseases": [
            {
              "disease":              str,
              "final_score":          float,   # mean(t1,t2), missing→0
              "symp_score":           float,   # tool1 original score
              "exam_score":           float,   # tool2 original score
              "softmax_prob":         float,
              "status_distribution":  {status: P(status|d), ...}  top5
            },
            ...
          ],
          "top_disease":        str,
          "top_disease_prob":   float,
          "top_status":         str,
          "top_status_prob":    float,
          "confidence_level":   "high" | "medium" | "low",
        }
    """
    t1_map = {
        c["disease"]: c
        for c in tool1_output.get("candidate_diseases", [])
    }
    t2_map = {
        c["disease"]: c
        for c in tool2_output.get("candidate_diseases", [])
    }

    all_diseases = set(t1_map) | set(t2_map)

    # -- Disease merging ------------------------------------
    merged = []
    for disease in all_diseases:
        s1 = t1_map[disease]["total_score"] if disease in t1_map else 0.0
        s2 = t2_map[disease]["total_score"] if disease in t2_map else 0.0
        final_score = (s1 + s2) / 2.0

        if final_score < min_score:
            continue

        # Status merging (per disease)
        d1_dist = t1_map[disease].get("status_distribution", {}) if disease in t1_map else {}
        d2_dist = t2_map[disease].get("status_distribution", {}) if disease in t2_map else {}
        merged_status = _merge_status_distribution(d1_dist, d2_dist, top_n=top_n)

        matched_symptoms = t1_map[disease].get("matched_symptoms", []) if disease in t1_map else []
        matched_exams    = t2_map[disease].get("matched_exams",    []) if disease in t2_map else []

        merged.append({
            "disease":             disease,
            "final_score":         round(final_score, 4),
            "symp_score":          round(s1, 4),
            "exam_score":          round(s2, 4),
            "status_distribution": merged_status,
            "matched_symptoms":    matched_symptoms,
            "matched_exams":       matched_exams,
        })

    if not merged:
        return {
            "candidate_diseases": [],
            "top_disease":        None,
            "top_disease_prob":   0.0,
            "top_status":         None,
            "top_status_prob":    0.0,
            "confidence_level":   "low",
        }

    # top_n → softmax
    merged.sort(key=lambda x: x["final_score"], reverse=True)
    merged = merged[:top_n]

    probs = _softmax([d["final_score"] for d in merged])
    for i, d in enumerate(merged):
        d["softmax_prob"] = probs[i]

    # -- Top-level summary ------------------------------------------
    top_disease      = merged[0]["disease"]
    top_disease_prob = merged[0]["softmax_prob"]

    # top_status comes from the status_distribution of top_disease
    top_status_dist  = merged[0].get("status_distribution", {})
    if top_status_dist:
        top_status      = max(top_status_dist, key=top_status_dist.get)
        top_status_prob = top_status_dist[top_status]
    else:
        top_status      = None
        top_status_prob = 0.0

    if top_disease_prob >= 0.6:
        confidence = "high"
    elif top_disease_prob >= 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    print(f"[Tool3] top_disease={top_disease}  prob={top_disease_prob}  confidence={confidence}")
    print(f"[Tool3] top_status={top_status}     prob={top_status_prob}")

    return {
        "candidate_diseases": merged,
        "top_disease":        top_disease,
        "top_disease_prob":   top_disease_prob,
        "top_status":         top_status,
        "top_status_prob":    top_status_prob,
        "confidence_level":   confidence,
    }


# ─────────────────────────────────────────────
# LangChain tool wrapper
# ─────────────────────────────────────────────

@tool
def combine_diagnosis(
    tool1_json: str,
    tool2_json: str,
) -> str:
    """
    Merge the outputs of symptom matching (Tool1) and examination matching (Tool2) into a combined diagnosis.

    Disease: mean(tool1_score, tool2_score) → top5 → softmax
    Status:  per disease, mean(tool1_prob, tool2_prob) → normalize → top5
    missing in one tool → score/prob = 0

    Args:
        tool1_json: JSON string output of Tool1 retrieve_by_symptoms
        tool2_json: JSON string output of Tool2 retrieve_by_examinations

    Returns:
        JSON string; each candidate disease contains final_score, softmax_prob, status_distribution
    """
    try:
        t1 = json.loads(tool1_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "tool1_json 解析失败"}, ensure_ascii=False)

    try:
        t2 = json.loads(tool2_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "tool2_json 解析失败"}, ensure_ascii=False)

    result = combine_results(t1, t2)
    return json.dumps(convert_numpy(result), ensure_ascii=False, indent=2)