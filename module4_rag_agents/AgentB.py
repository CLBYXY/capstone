"""
Agent B - disease-specific query
Handles intent B: after describing many symptoms/examinations, the user asks "do I have disease xxx?".

Pipeline:
  Step1: intent detection (LLM) - decide whether the user asks about a specific disease and extract disease_mentioned
  Step2: KG disease matching - compare with all Disease nodes using SapBERT and return the most similar standard name
  Step3a: build a local symptom vector space limited to disease_target (FAISS subset) [first turn only, cached]
  Step3b: user symptoms -> vector retrieval + scoring in the local space (identical to Tool1) [first turn only, cached]
  Step3c: user examinations -> reuse Tool2 logic and take the score of disease_target [rerun every turn, exam_input accumulates]
  Step3d: mean_score = (symp_score + exam_score) / 2, merge status_distribution
  Step4:  output decision
          confirmed  -> LLM rewriting -> session ends
          further    -> LLM rewriting -> Tool4 recommendation -> ask to add exams -> rerun Step3c+3d
          no_disease -> LLM rewriting -> session ends
          mean_score < 0.5 -> fall back to Agent A

Exit conditions of the multi-turn loop:
  - top_status becomes confirmed or no_disease
  - the user declines to add examination results
  - more than the maximum of MAX_TURNS turns

Note: if intent detection finds no disease question, the request goes directly to Agent A.
"""

import os
import json
import re
import numpy as np
from typing import Any
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

from Agent1_tool1 import (
    extract_symptoms,
    vector_search_symptom,
    score_diseases,
    get_status_counts,
    compute_sf_idf,
    _get_embeddings,
    load_symptom_vector_index,
    build_symptom_vector_index,
    convert_numpy as t1_convert,
)
from Agent1_tool2 import (
    exam_matching_pipeline,
    load_exam_vector_index,
    build_exam_vector_index,
    convert_numpy as t2_convert,
)
from Agent2_tool3 import _merge_status_distribution
from Agent2_tool4 import recommend_exams, convert_numpy as t4_convert
from AgentA import (
    run_agent_a,
    polish_exam_output,
    split_input,
    ask_yes_no,
    ask_exam_results,
)


# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────

NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://<your-instance>.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

CONFIRMED_STATUS   = "diagnosed at the time of the visit"
FURTHER_STATUS     = "further examination required"
NO_DISEASE         = "no disease"
FALLBACK_THRESHOLD = 0.5   # mean_score below this value -> fall back to Agent A
MAX_TURNS          = 5     # maximum number of turns


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
# 2. Step 1 - intent detection (LLM)
# ─────────────────────────────────────────────

INTENT_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a medical intent detection assistant.
Analyze the patient's message and determine whether they are specifically asking about a particular disease.

This includes:
- Direct questions: "Is this diabetes?", "Could this be pneumonia?", "Do I have Crohn's disease?"
- Implicit speculation: "I've always assumed this was gastritis", "I've been told it might be lupus", "Could what I have be COPD?"
- Comparisons or worries: "I'm worried this could be cancer", "Could this be related to heart disease?"

Rules:
1. Only set has_disease_query to true if the user is clearly asking about or speculating on a SPECIFIC named disease.
2. Extract the disease name EXACTLY as mentioned by the user (do not normalize it — return the original phrasing).
3. If multiple diseases are mentioned, extract the one the user seems most focused on (usually the last one or the one with a direct question).
4. Output ONLY valid JSON, no markdown.

Input: {user_input}

Output:
{{
  "has_disease_query": <true|false>,
  "disease_mentioned": "<disease name as written by user, or null>"
}}
"""
)


def detect_intent(user_input: str, llm: ChatOpenAI) -> dict:
    """
    Decide whether the user asks about a specific disease.

    Returns:
        {"has_disease_query": bool, "disease_mentioned": str | None}
    """
    raw = (INTENT_PROMPT | llm).invoke({"user_input": user_input}).content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        result = json.loads(raw)
        return {
            "has_disease_query": bool(result.get("has_disease_query", False)),
            "disease_mentioned": result.get("disease_mentioned"),
        }
    except json.JSONDecodeError:
        print(f"[detect_intent] JSON解析失败，原始输出：{raw}")
        return {"has_disease_query": False, "disease_mentioned": None}


# ─────────────────────────────────────────────
# 3. Step 2 - KG disease matching (SapBERT cosine similarity)
# ─────────────────────────────────────────────

def match_disease_to_kg(
    disease_mentioned: str,
    embeddings: HuggingFaceEmbeddings,
) -> str:
    """
    Align the disease name mentioned by the user to the standard Disease node name in the KG using SapBERT.

    Returns:
        disease_target: the standard disease name in the KG with the highest similarity
    """
    rows = neo4j_client.query("MATCH (d:Disease) RETURN d.name AS name")
    if not rows:
        raise ValueError("KG 中无 Disease 节点。")

    kg_names = [r["name"] for r in rows]

    from sentence_transformers import util as st_util
    model     = embeddings.client
    query_emb = model.encode(disease_mentioned, normalize_embeddings=True)
    kg_embs   = model.encode(kg_names, normalize_embeddings=True)
    sims      = st_util.cos_sim(query_emb, kg_embs)[0].tolist()

    best_idx      = int(np.argmax(sims))
    disease_target = kg_names[best_idx]
    best_sim      = round(float(sims[best_idx]), 4)

    print(f"[Step2] 用户疾病词='{disease_mentioned}' → KG标准名='{disease_target}'  sim={best_sim}")
    return disease_target


# ─────────────────────────────────────────────
# 4. Step 3a - build the local symptom vector space
# ─────────────────────────────────────────────

def build_disease_scoped_symptom_index(
    disease_target: str,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS | None:
    """
    Take only the Symptom nodes linked to disease_target and build a local FAISS index.
    sf_idf_scores in the metadata only contains the normalised SF-IDF of disease_target,
    compatible with the Tool1 global index format, so that score_diseases() can be reused directly.

    Returns:
        local FAISS index; None if the disease has no symptom nodes
    """
    rows = neo4j_client.query("""
        MATCH (d:Disease)-[:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
              -[:HAS_SYMPTOM]->(s:Symptom)
        WHERE d.name = $disease
        RETURN DISTINCT s.name AS name, elementId(s) AS neo4j_id
    """, {"disease": disease_target})

    if not rows:
        print(f"[Step3a] disease='{disease_target}' 在 KG 中无关联症状节点。")
        return None

    # Take the value of this disease from the global SF-IDF (already normalised)
    all_sf_idf = compute_sf_idf()

    texts     = [r["name"] for r in rows]
    metadatas = []
    for r in rows:
        # normalised SF-IDF of this symptom node for disease_target
        # the key must be "sf_idf_scores", consistent with how Tool1 v2 vector_search_symptom reads it
        sfidf_target = all_sf_idf.get(str(r["neo4j_id"]), {}).get(disease_target, 0.0)
        metadatas.append({
            "neo4j_id":      r["neo4j_id"],
            "name":          r["name"],
            "sf_idf_scores": json.dumps({disease_target: round(sfidf_target, 6)}),
        })

    scoped_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    print(f"[Step3a] disease='{disease_target}' 局部空间构建完成，共 {len(texts)} 个症状节点。")
    return scoped_store


# ─────────────────────────────────────────────
# 5. Step 3b - symptom scoring (reuses Tool1 logic)
# ─────────────────────────────────────────────

def compute_symptom_score_for_target(
    symptom_input:  str,
    disease_target: str,
    scoped_store:   FAISS,
    llm:            ChatOpenAI,
    embeddings:     HuggingFaceEmbeddings,
) -> tuple[float, dict, list]:
    """
    Score disease_target on symptoms in the local symptom vector space.
    The scoring formula is identical to Tool1:
        contribution = 0.5 * final_sim + 0.5 * P(disease_target | symp)
        symp_score   = sum of top5 contributions

    Returns:
        (symp_score, status_distribution, matched_symptoms, extracted_symptoms)
    """
    # Step 1: LLM extracts symptoms (reuses Tool1)
    extracted = extract_symptoms(symptom_input, llm)
    for s in extracted:
        print(f"[Step3b|extract] '{s['core']}'  duration={s['duration']}")

    # Step 2: search in the local space (reuses Tool1 vector_search_symptom)
    all_candidates = []
    for sym in extracted:
        candidates = vector_search_symptom(
            sym["core"],
            sym.get("duration", {"value": None, "unit": None, "duration_raw": None}),
            scoped_store,
            embeddings,
        )
        strategy = "diff" if sym["duration"].get("value") is not None else "embed"
        for c in candidates:
            print(
                f"[Step3b|{strategy}] '{sym['core']}' → '{c['kg_name']}'  "
                f"syp={c['syp_sim']}  best_time={c['best_time_sim']}  final={c['final_sim']}"
            )
        all_candidates.extend(candidates)

    # Step 3: scoring (reuses Tool1 score_diseases)
    results = score_diseases(all_candidates)

    for d in results:
        if d["disease"] == disease_target:
            print(
                f"[Step3b] disease='{disease_target}'  "
                f"symp_score={d['total_score']}  status={d['status_distribution']}"
            )
            return (
                float(d["total_score"]),
                d.get("status_distribution", {}),
                d.get("matched_symptoms", []),
                extracted,                          # passed through to the format_output fallback
            )

    print(f"[Step3b] disease='{disease_target}' 未出现在症状打分结果中，symp_score=0.0")
    return 0.0, {}, [], extracted


# ─────────────────────────────────────────────
# 6. Step 3c - examination scoring (reuses Tool2 logic)
# ─────────────────────────────────────────────

def compute_exam_score_for_target(
    exam_input:     str,
    disease_target: str,
    exam_store:     FAISS,
    llm:            ChatOpenAI,
    embeddings:     HuggingFaceEmbeddings,
) -> tuple[float, dict, list]:
    """
    Fully reuses Tool2 exam_matching_pipeline(),
    takes the score and status_distribution of disease_target from the full result.

    Returns:
        (exam_score, status_distribution, matched_exams, t2_raw)
    """
    if not exam_input or not exam_input.strip():
        print("[Step3c] 无检查输入，exam_score=0.0")
        return 0.0, {}, [], {}

    t2_raw = exam_matching_pipeline(exam_input, llm, exam_store, embeddings)

    for d in t2_raw.get("candidate_diseases", []):
        if d["disease"] == disease_target:
            print(
                f"[Step3c] disease='{disease_target}'  "
                f"exam_score={d['total_score']}  status={d['status_distribution']}"
            )
            return (
                float(d["total_score"]),
                d.get("status_distribution", {}),
                d.get("matched_exams", []),
                t2_raw,
            )

    print(f"[Step3c] disease='{disease_target}' 未出现在检查打分结果中，exam_score=0.0")
    return 0.0, {}, [], t2_raw


# ─────────────────────────────────────────────
# 7. Step 3d - merge scores and status
# ─────────────────────────────────────────────

def merge_scores(
    symp_score:   float,
    exam_score:   float,
    symp_status:  dict,
    exam_status:  dict,
) -> tuple[float, dict, str, float]:
    """
    Merge the scores and status_distribution of symptoms and examinations.

    Returns:
        (mean_score, merged_status_dist, top_status, top_status_prob)
    """
    mean_score = round((symp_score + exam_score) / 2.0, 4)

    merged_status = _merge_status_distribution(symp_status, exam_status)

    if merged_status:
        top_status      = max(merged_status, key=merged_status.get)
        top_status_prob = merged_status[top_status]
    else:
        top_status      = NO_DISEASE
        top_status_prob = 0.0

    print(
        f"[Step3d] symp_score={symp_score}  exam_score={exam_score}  "
        f"mean_score={mean_score}  top_status={top_status}  "
        f"top_status_prob={top_status_prob}"
    )
    return mean_score, merged_status, top_status, top_status_prob


# ─────────────────────────────────────────────
# 8. Step 4 - output decision
# ─────────────────────────────────────────────

POLISH_PROMPT_B = PromptTemplate(
    input_variables=[
        "disease_target", "mean_score", "top_status",
        "conclusion_raw", "matched_symptoms", "matched_exams"
    ],
    template="""
你是一位经验丰富的临床医生，正在门诊向患者口头解释评估结果。
请根据下方数据，用自然流畅的中文向患者说明情况，语气平和、专业、有温度。

【写作要求】
1. 下方"匹配症状"和"匹配检查"是系统从知识图谱中检索到的原始词条，
   同一个症状可能以多种表述出现（如"口干""口渴增加""持续口渴（多饮）""过度口渴"
   本质上是同一个症状）。你需要先在心里做归纳，再用1-3个临床术语表达，
   绝对不要把原始词条逐一照抄到输出中。
2. 用完整的自然语句描述，像医生和患者说话一样，而不是列举术语清单。
   好的示例：
     "您描述的多饮多尿、近期体重下降，以及视物模糊等表现，与糖尿病的典型症状较为吻合。"
   不好的示例（禁止）：
     "与该疾病相符的症状包括：口干、持续口渴（多饮）、口干和口渴、口渴增加……"
3. 检查结果同理，提炼出关键异常指标，用一句话说明其临床意义，不要罗列所有检查名称。
4. 禁止编造下方数据中未出现的症状、检查项目或结论。
5. 禁止给出用药或治疗建议。
6. 不使用 Markdown、列表符号或标题，只输出纯文本。

【输出结构】严格按以下顺序，共输出3-4句话：
A. 总结句：基于原始结论，用一句话告知患者整体评估结果。
B. 症状句（仅当匹配症状不为"（无）"时）：
   将症状词条归纳为1-3个核心症状群，用一句完整自然的话描述，
   点明这些表现与{disease_target}的关联。
C. 检查句（仅当匹配检查不为"（无）"时）：
   提炼关键异常指标，用一句话说明检查结果与该疾病的关系。
D. 收尾句：根据 status 给出适当的后续说明：
   - confirmed  → 建议尽快正式就诊，由医生进行确诊和后续管理。
   - further    → 说明目前依据尚不充分，建议进一步完善检查。
   - no_disease → 简短宽慰，如症状持续建议就诊排查其他原因。

--- 评估数据 ---
询问疾病：{disease_target}
匹配得分：{mean_score}
状态结论：{top_status}
原始结论：{conclusion_raw}

--- 匹配症状（知识图谱原始词条，请归纳后使用，勿照搬）---
{matched_symptoms}

--- 匹配检查（知识图谱原始词条，请提炼后使用，勿照搬）---
{matched_exams}
---
"""
)


def format_output(
    disease_target:   str,
    mean_score:       float,
    top_status:       str,
    top_status_prob:  float,
    matched_symptoms: list,
    matched_exams:    list,
    llm:              ChatOpenAI,
    extracted_symptoms: list | None = None,   # Tool1 extracted detail (for fallback)
    t2_raw:             dict | None = None,   # Tool2 full output (for fallback)
) -> tuple[str, str]:
    """
    Generate a structured conclusion from top_status, then rewrite it with the LLM.
    matched_symptoms / matched_exams prefer the results of the scoring stage;
    if empty, use extracted_symptoms (symptom terms extracted by the LLM) and t2_raw (raw Tool2 output)
    two-level fallback, aligned with the polish_output() logic of Agent A.

    Returns:
        (raw_conclusion, polished_conclusion)
    """
    if top_status == CONFIRMED_STATUS:
        raw = (
            f"根据您的描述，症状和检查结果与【{disease_target}】高度吻合"
            f"（匹配得分 {mean_score:.2f}，置信概率 {top_status_prob:.0%}）。"
        )
    elif top_status == FURTHER_STATUS:
        raw = (
            f"目前信息提示存在【{disease_target}】的可能"
            f"（匹配得分 {mean_score:.2f}，进一步检查概率 {top_status_prob:.0%}），"
            f"建议完善相关检查以明确诊断。"
        )
    else:
        raw = (
            f"根据您的描述，【{disease_target}】的可能性较低"
            f"（匹配得分 {mean_score:.2f}，无病概率 {top_status_prob:.0%}）。"
        )

    # -- Symptom details: three-level fallback ----------------------------
    # 1. matched_symptoms returned by score_diseases() (KG standard names)
    symp_entries = list(dict.fromkeys(matched_symptoms))   # de-duplicate while keeping order

    # 2. Use the core terms extracted by the LLM from extracted_symptoms (covers cases where KG names differ)
    if not symp_entries and extracted_symptoms:
        symp_entries = list(dict.fromkeys(
            s["core"] for s in extracted_symptoms if s.get("core")
        ))

    symp_str = "\n".join(f"- {s}" for s in symp_entries) if symp_entries else "（无）"

    # -- Examination details: three-level fallback ------------------------
    # 1. matched_exams returned by score_diseases() (format "type:result")
    exam_entries = list(dict.fromkeys(matched_exams))

    # 2. Take matched_exams of disease_target from t2_raw candidate_diseases
    if not exam_entries and t2_raw:
        for d in t2_raw.get("candidate_diseases", []):
            if d["disease"] == disease_target:
                exam_entries = list(dict.fromkeys(d.get("matched_exams", [])))
                break

    # 3. Take the top-level matched entries from t2_raw exam_candidates
    if not exam_entries and t2_raw:
        for detail in t2_raw.get("exam_candidates", []):
            cands = detail.get("candidates", [])
            if cands:
                kt = cands[0].get("kg_type_name", "")
                kr = cands[0].get("result_name", "")
                entry = f"{kt}" + (f"（{kr}）" if kr else "")
                if kt and entry not in exam_entries:
                    exam_entries.append(entry)

    exam_str = "\n".join(f"- {e}" for e in exam_entries) if exam_entries else "（无）"

    print(f"[polish] matched_symptoms → {symp_entries}")
    print(f"[polish] matched_exams    → {exam_entries}")

    try:
        polished = (POLISH_PROMPT_B | llm).invoke({
            "disease_target":    disease_target,
            "mean_score":        f"{mean_score:.4f}",
            "top_status":        top_status,
            "conclusion_raw":    raw,
            "matched_symptoms":  symp_str,
            "matched_exams":     exam_str,
        }).content.strip()
    except Exception as e:
        print(f"[polish] LLM润色失败，使用原始输出：{e}")
        polished = raw

    return raw, polished


# ─────────────────────────────────────────────
# 9. Main entry run_agent_b
# ─────────────────────────────────────────────

def run_agent_b(
    user_input:    str,
    llm:           ChatOpenAI,
    symptom_store: FAISS,            # Tool1 global index (used for the Agent A fallback)
    exam_store:    FAISS,            # Tool2 global index
    embeddings:    HuggingFaceEmbeddings,
    verbose:       bool = True,
) -> dict[str, Any]:
    """
    Main function of Agent B, supporting a multi-turn loop.

    The first turn performs intent detection, KG matching, local symptom space construction and symptom scoring (Step1~3b);
    Results are cached in session_state; later turns only rerun Step3c (examination scoring) + Step3d (merging).

    Multi-turn loop condition: top_status = further -> Tool4 recommendation -> ask to add exams -> rerun Step3c+3d
    Exit conditions:
      - top_status becomes confirmed or no_disease
      - the user declines to add examination results
      - more than MAX_TURNS turns

    Args:
        user_input:     full patient input (symptom description + disease question)
        llm:            OpenAI LLM instance
        symptom_store:  Tool1 global FAISS index (only used for the fallback)
        exam_store:     Tool2 global FAISS index
        embeddings:     SapBERT embeddings
        verbose:        whether to print intermediate logs

    Returns:
        {
          "mode":            "agent_b" | "fallback_agent_a",
          "disease_target":  str,
          "turn":            int,          # number of turns actually executed
          "mean_score":      float,
          "top_status":      str,
          "output_raw":      str,
          "output_polished": str,
          "t4_result":       dict | None,  # set when the last turn is "further"
          "history":         list[dict],   # output record of each turn
        }
    """
    print("\n" + "=" * 60)
    print("Agent B 开始（意图B：定向疾病查询）")
    print("=" * 60)

    # -- Step 1: intent detection (once) -----------------------------
    if verbose:
        print("\n[Step1] 意图识别中...")
    intent = detect_intent(user_input, llm)
    print(f"[Step1] has_disease_query={intent['has_disease_query']}  "
          f"disease_mentioned='{intent['disease_mentioned']}'")

    if not intent["has_disease_query"] or not intent["disease_mentioned"]:
        print("[Agent B] 未检测到特定疾病询问，转入 Agent A。")
        run_agent_a(user_input, llm, symptom_store, exam_store, embeddings, verbose)
        return {"mode": "fallback_agent_a", "reason": "no_disease_intent"}

    disease_mentioned = intent["disease_mentioned"]

    # -- Step 2: KG disease matching (once) -------------------------
    if verbose:
        print(f"\n[Step2] KG 疾病匹配：'{disease_mentioned}'")
    disease_target = match_disease_to_kg(disease_mentioned, embeddings)

    # -- Split the user input (once) --------------------------------
    if verbose:
        print("\n[input split] 拆分用户输入（症状 / 检查）...")
    symptom_input, exam_input = split_input(user_input, llm)
    if verbose:
        print(f"  症状：{symptom_input}")
        print(f"  检查：{exam_input or '（无）'}")

    # -- Step 3a: build the local symptom vector space (once) --------
    if verbose:
        print(f"\n[Step3a] 构建 '{disease_target}' 局部症状空间...")
    scoped_store = build_disease_scoped_symptom_index(disease_target, embeddings)

    if scoped_store is None:
        print(f"[Agent B] '{disease_target}' 无症状节点，回退 Agent A。")
        run_agent_a(user_input, llm, symptom_store, exam_store, embeddings, verbose)
        return {
            "mode":           "fallback_agent_a",
            "reason":         "no_symptom_nodes_in_kg",
            "disease_target": disease_target,
        }

    # -- Step 3b: symptom scoring (once; symptoms do not change) -----
    if verbose:
        print(f"\n[Step3b] 症状打分（disease_target='{disease_target}'）...")
    symp_score, symp_status, matched_symptoms, extracted_symptoms = \
        compute_symptom_score_for_target(
            symptom_input, disease_target, scoped_store, llm, embeddings
        )

    # -- session_state initialisation ----------------------------------
    session_state = {
        "disease_target":    disease_target,
        "exam_input":        exam_input,        # accumulated new examination descriptions
        "done_exams":        [],                # examination types already done (excluded by Tool4)
        "symp_score":        symp_score,
        "symp_status":       symp_status,
        "matched_symptoms":  matched_symptoms,
        "extracted_symptoms": extracted_symptoms,
        "turn":              0,
        "t4_result":         None,
        "history":           [],
    }

    # ══════════════════════════════════════
    # Main multi-turn loop
    # ══════════════════════════════════════
    while session_state["turn"] < MAX_TURNS:
        turn = session_state["turn"]
        print(f"\n─── 第 {turn + 1} 轮 ───────────────────────────────────────")

        # -- Step 3c: examination scoring (rerun every turn; exam_input accumulates) --
        if verbose:
            print(f"\n[Step3c] 检查打分（turn={turn + 1}，"
                  f"exam_input='{session_state['exam_input'] or '（无）'}'）...")
        exam_score, exam_status, matched_exams, t2_raw = compute_exam_score_for_target(
            session_state["exam_input"], disease_target, exam_store, llm, embeddings
        )

        # -- Step 3d: merge -------------------------------------
        mean_score, merged_status, top_status, top_status_prob = merge_scores(
            session_state["symp_score"], exam_score,
            session_state["symp_status"], exam_status,
        )

        # -- Low-score fallback (first turn only; later turns only add symptoms/exams, so no fallback)
        if turn == 0 and mean_score < FALLBACK_THRESHOLD:
            print(
                f"[Agent B] mean_score={mean_score} < {FALLBACK_THRESHOLD}，"
                f"'{disease_target}' 匹配度不足，回退 Agent A。"
            )
            fallback_notice = (
                f"根据您的描述，暂未发现与【{disease_target}】的明显关联"
                f"（匹配得分 {mean_score:.2f}），将为您进行开放式诊断分析。"
            )
            print(f"\n[提示] {fallback_notice}")
            run_agent_a(user_input, llm, symptom_store, exam_store, embeddings, verbose)
            return {
                "mode":           "fallback_agent_a",
                "reason":         "low_score",
                "disease_target": disease_target,
                "mean_score":     mean_score,
            }

        # -- Step 4: output decision + LLM rewriting ---------------------
        if verbose:
            print(f"\n[Step4] 输出决策：top_status='{top_status}'")
        raw_output, polished_output = format_output(
            disease_target, mean_score, top_status, top_status_prob,
            matched_symptoms if turn == 0 else session_state["matched_symptoms"],
            matched_exams,
            llm,
            extracted_symptoms=session_state["extracted_symptoms"],
            t2_raw=t2_raw,
        )

        print(f"\n[诊断结果]\n{polished_output}")

        # record the result of this turn
        turn_record = {
            "turn":            turn + 1,
            "mean_score":         mean_score,
            "exam_score":         exam_score,
            "top_status":         top_status,
            "top_status_prob":    top_status_prob,
            "status_distribution": merged_status,
            "output_raw":         raw_output,
            "output_polished":    polished_output,
            "matched_exams":      matched_exams,
            "t4_result":          None,
        }

        # -- Exit condition: confirmed or no_disease ---------------
        if top_status in (CONFIRMED_STATUS, NO_DISEASE):
            print(f"\n[Agent B] top_status='{top_status}'，诊断已明确，会话结束。")
            session_state["history"].append(turn_record)
            break

        # -- "further" branch ---------------------------------------
        # Tool4: examination recommendation
        if not ask_yes_no("是否需要进一步检查建议？"):
            print("[Agent B] 用户选择结束。")
            session_state["history"].append(turn_record)
            break

        # build the Tool4 input (only disease_target)
        tool4_input = [{
            "disease":             disease_target,
            "final_score":         mean_score,
            "symp_score":          session_state["symp_score"],
            "exam_score":          exam_score,
            "softmax_prob":        1.0,
            "status_distribution": merged_status,
            "matched_symptoms":    session_state["matched_symptoms"],
            "matched_exams":       matched_exams,
        }]

        # done_exams: accumulate completed examinations so that Tool4 does not recommend them again
        current_done = list({
            e.split(":")[0].strip() for e in matched_exams
        }) if matched_exams else []
        session_state["done_exams"] = list(
            set(session_state["done_exams"]) | set(current_done)
        )

        if verbose:
            print(f"\n[Tool4] 推荐检查（disease='{disease_target}'，"
                  f"done={session_state['done_exams']}）...")
        t4_result = recommend_exams(tool4_input, session_state["done_exams"])
        t4_result = t4_convert(t4_result)

        global_recs = t4_result.get("global_recommendations", [])
        per_d_recs  = t4_result.get("per_disease_recommendations", [])
        polished_exam = polish_exam_output(global_recs, per_d_recs, llm)
        print(f"\n[推荐检查]\n{polished_exam}")

        turn_record["t4_result"] = t4_result
        session_state["t4_result"] = t4_result
        session_state["history"].append(turn_record)

        # ask whether to add examination results
        if not ask_yes_no("是否补充检查结果以进一步评估？"):
            print("[Agent B] 用户选择不补充，会话结束。")
            break

        new_exam_desc, new_done = ask_exam_results()
        if new_exam_desc:
            # append to exam_input; the next Step3c re-parses all of it
            if session_state["exam_input"]:
                session_state["exam_input"] += "。" + new_exam_desc
            else:
                session_state["exam_input"] = new_exam_desc
        session_state["done_exams"].extend(new_done)
        session_state["turn"] += 1

    else:
        print(f"\n[Agent B] 已达最大轮次 {MAX_TURNS}，会话结束。")

    print("\n" + "=" * 60)
    print("Agent B 会话结束")
    print("=" * 60)

    # use the result of the last turn as the top-level return fields
    last = session_state["history"][-1] if session_state["history"] else {}
    # merged_status is not stored in the last turn_record; rebuild it from the end of history
    # for simplicity, use top_status_prob and top_status from last; the full distribution is kept in history
    last_status_dist = last.get("status_distribution", {})
    return {
        "mode":               "agent_b",
        "disease_target":     disease_target,
        "turn":               session_state["turn"] + 1,
        "mean_score":         last.get("mean_score", 0.0),
        "symp_score":         session_state["symp_score"],
        "exam_score":         last.get("exam_score", 0.0),
        "top_status":         last.get("top_status", ""),
        "top_status_prob":    last.get("top_status_prob", 0.0),
        "status_distribution": last_status_dist,
        "matched_symptoms":   session_state["matched_symptoms"],
        "matched_exams":      last.get("matched_exams", []),
        "output_raw":         last.get("output_raw", ""),
        "output_polished":    last.get("output_polished", ""),
        "t4_result":          session_state["t4_result"],
        "history":            session_state["history"],
    }


# ─────────────────────────────────────────────
# 10. Initialisation & entry point
# ─────────────────────────────────────────────

def init_agent_b(
    openai_api_key:      str,
    symptom_index_path:  str  = "symptom_faiss_index",
    exam_index_path:     str  = "exam_faiss_index",
    rebuild_index:       bool = False,
) -> tuple[ChatOpenAI, FAISS, FAISS, HuggingFaceEmbeddings]:
    """
    Initialise the LLM, vector indexes and embeddings needed by Agent B.

    Args:
        rebuild_index: True -> rebuild the FAISS index (use on the first run or after the KG is updated)
                       False -> load the existing index directly (normal run)

    Returns:
        (llm, symptom_store, exam_store, embeddings)
    """
    llm        = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
    embeddings = _get_embeddings()

    if rebuild_index:
        print("[init] 重新构建 symptom 索引...")
        symp_store = build_symptom_vector_index(symptom_index_path)
        print("[init] 重新构建 exam 索引...")
        exam_store = build_exam_vector_index(exam_index_path)
    else:
        symp_store = load_symptom_vector_index(symptom_index_path)
        exam_store = load_exam_vector_index(exam_index_path)

    print("[init] Agent B 初始化完成。")
    return llm, symp_store, exam_store, embeddings