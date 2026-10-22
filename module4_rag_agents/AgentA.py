"""
Agent A - open-ended diagnostic dialogue
Supports intent A (open-ended question) and intent C (multi-turn addition of examination results).

Call chain:
  Tool1 (symptom matching) + Tool2 (examination matching) -> Tool3 (combined diagnosis) -> output tiers
  -> ask whether to assess further -> Tool4 (recommend examinations) -> ask to add results -> loop

Output tier rules (in priority order; one disease can match several):
  Rule 0: diseases whose top_status is no_disease are skipped
  Rule 1: highest confirmed score < 2.0 and no "further" -> no_disease (fallback)
  Rule 2: top1 + confirmed -> most_likely
  Rule 3: top2-5 + confirmed + score > 3 -> possible
  Rule 4: top5 + further -> at_risk

Exit conditions:
  - the user declines further assessment
  - the user declines to add examination results
  - confidence=high and top_status=confirmed/no_disease
"""

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from Agent1_tool1 import (
    symptom_matching_pipeline,
    convert_numpy as t1_convert,
    _get_embeddings,
    load_symptom_vector_index,
    build_symptom_vector_index,
)
from Agent1_tool2 import (
    exam_matching_pipeline,
    convert_numpy as t2_convert,
    load_exam_vector_index,
    build_exam_vector_index,
)
from Agent2_tool3 import combine_results, convert_numpy as t3_convert
from Agent2_tool4 import recommend_exams, convert_numpy as t4_convert


# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────

CONFIRMED_STATUS           = "diagnosed at the time of the visit"
FURTHER_STATUS             = "further examination required"
NO_DISEASE                 = "no disease"
SCORE_THRESHOLD            = 3.0   # confirmed + score > 3 is required for "possible"
NO_DISEASE_SCORE_THRESHOLD = 2.0   # if the highest confirmed score is below this value and there is no "further" -> no_disease


# ─────────────────────────────────────────────
# 1. Output tiers
# ─────────────────────────────────────────────

def classify_output(candidate_diseases: list[dict]) -> dict[str, list]:
    """
    Group the top-5 candidate diseases into tiers by rules (in priority order).

    Returns:
        {
          "most_likely": [d],
          "possible":    [d, ...],
          "at_risk":     [d, ...],
          "no_disease":  bool,
        }
    """
    result = {"most_likely": [], "possible": [], "at_risk": [], "no_disease": False}

    if not candidate_diseases:
        return result

    # Rule 0: exclude diseases whose top_status is no_disease
    active = []
    for d in candidate_diseases:
        dist       = d.get("status_distribution", {})
        top_status = max(dist, key=dist.get) if dist else None
        if top_status == NO_DISEASE:
            continue
        active.append((d, top_status))

    if not active:
        result["no_disease"] = True
        return result

    # Rule 1 (fallback, checked first): highest confirmed score < threshold and no "further" -> no_disease
    confirmed_scores = [
        d.get("final_score", 0.0) for d, st in active if st == CONFIRMED_STATUS
    ]
    has_further = any(st == FURTHER_STATUS for _, st in active)
    max_confirmed = max(confirmed_scores) if confirmed_scores else 0.0

    if max_confirmed < NO_DISEASE_SCORE_THRESHOLD and not has_further:
        result["no_disease"] = True
        return result

    # Rules 2/3/4
    for i, (d, top_status) in enumerate(active):
        score = d.get("final_score", 0.0)
        if i == 0 and top_status == CONFIRMED_STATUS:
            result["most_likely"].append(d)
        if top_status == CONFIRMED_STATUS and score > SCORE_THRESHOLD and i > 0:
            result["possible"].append(d)
        if top_status == FURTHER_STATUS:
            result["at_risk"].append(d)

    return result


def format_output_message(classified: dict[str, list]) -> str:
    lines = []
    if classified["most_likely"]:
        d    = classified["most_likely"][0]
        prob = d.get("softmax_prob", 0)
        lines.append(f"根据您的描述，您最有可能有【{d['disease']}】（概率 {prob:.0%}）。")
    if classified["possible"]:
        names = "、".join(d["disease"] for d in classified["possible"])
        lines.append(f"也有可能：{names}。")
    if classified["at_risk"]:
        names = "、".join(d["disease"] for d in classified["at_risk"])
        lines.append(f"另外，您有以下疾病的风险，建议进一步排查：{names}。")
    if not lines and classified["no_disease"]:
        lines.append("根据您当前提供的信息，暂未发现明显的疾病指征。")
    return "\n".join(lines) if lines else "暂时无法根据当前信息给出明确结论，建议补充更多检查结果。"


# ─────────────────────────────────────────────
# 2. LLM rewriting - diagnosis result
# ─────────────────────────────────────────────

POLISH_PROMPT = PromptTemplate(
    input_variables=["structured_output", "retrieved_symptoms", "retrieved_exams"],
    template="""
You are a professional medical assistant helping to communicate diagnostic results to a patient.
Your task is to rephrase the structured diagnostic summary below into clear, professional,
and empathetic language in Chinese.

STRICT RULES — you must follow all of these:
1. Do NOT invent, add, or infer any disease, symptom, examination, or conclusion not explicitly listed below.
2. Do NOT provide medical advice, treatment recommendations, or drug suggestions.
3. Do NOT use alarming or emotionally charged language.
4. Keep all disease names, examination names, and probability values exactly as given.
5. You may only reorder, connect, or rephrase the information already present.
6. Output plain text only — no markdown, no bullet points, no headers.
7. Write in Chinese (Simplified).

--- Structured diagnostic summary ---
{structured_output}

--- Matched symptoms from knowledge graph ---
{retrieved_symptoms}

--- Matched examinations from knowledge graph ---
{retrieved_exams}
---

Rephrase the structured summary into 2-4 natural sentences. Stay strictly within the provided data.
"""
)


def polish_output(
    structured_output: str,
    t3_result:         dict,
    llm:               ChatOpenAI,
    t1_raw:            dict | None = None,
    t2_raw:            dict | None = None,
) -> str:
    """
    LLM rewriting of the diagnosis output.
    Taken from t3_result.candidate_diseases[*].matched_symptoms/matched_exams first.
    The three-level fallback ensures that the LLM always receives content.
    """
    top_diseases = t3_result.get("candidate_diseases", [])

    # -- Symptoms --------------------------------------------------
    symp_entries, seen_symps = [], set()

    for d in top_diseases:
        for s in d.get("matched_symptoms", []):
            if s not in seen_symps:
                symp_entries.append(f"- {s}")
                seen_symps.add(s)

    if not symp_entries and t1_raw:
        for d in t1_raw.get("candidate_diseases", []):
            for s in d.get("matched_symptoms", []):
                if s not in seen_symps:
                    symp_entries.append(f"- {s}")
                    seen_symps.add(s)

    if not symp_entries and t1_raw:
        for detail in t1_raw.get("symptom_candidates", []):
            cands = detail.get("candidates", [])
            if cands:
                kg = cands[0].get("kg_name", "")
                if kg and kg not in seen_symps:
                    symp_entries.append(f"- {kg}")
                    seen_symps.add(kg)

    retrieved_symptoms = "\n".join(symp_entries) if symp_entries else "（无）"

    # -- Examinations --------------------------------------------------
    exam_entries, seen_exams = [], set()

    for d in top_diseases:
        for e in d.get("matched_exams", []):
            if e not in seen_exams:
                exam_entries.append(f"- {e}")
                seen_exams.add(e)

    if not exam_entries and t2_raw:
        for d in t2_raw.get("candidate_diseases", []):
            for e in d.get("matched_exams", []):
                if e not in seen_exams:
                    exam_entries.append(f"- {e}")
                    seen_exams.add(e)

    if not exam_entries and t2_raw:
        for detail in t2_raw.get("exam_candidates", []):
            cands = detail.get("candidates", [])
            if cands:
                kt    = cands[0].get("kg_type_name", "")
                kr    = cands[0].get("result_name", "")
                entry = f"- {kt}" + (f" / {kr}" if kr else "")
                if kt and entry not in seen_exams:
                    exam_entries.append(entry)
                    seen_exams.add(entry)

    retrieved_exams = "\n".join(exam_entries) if exam_entries else "（无）"

    try:
        return (POLISH_PROMPT | llm).invoke({
            "structured_output":  structured_output,
            "retrieved_symptoms": retrieved_symptoms,
            "retrieved_exams":    retrieved_exams,
        }).content.strip()
    except Exception as e:
        print(f"[polish] LLM润色失败，使用原始输出：{e}")
        return structured_output


# ─────────────────────────────────────────────
# 3. LLM rewriting - examination recommendation (adapted to Tool4 v2 output)
# ─────────────────────────────────────────────

POLISH_EXAM_PROMPT = PromptTemplate(
    input_variables=["global_recs", "per_disease_recs"],
    template="""
You are a professional medical assistant.
Rephrase the following examination recommendations into clear, professional Chinese.

STRICT RULES:
1. Do NOT add any exam not listed below.
2. Do NOT provide treatment advice or drug recommendations.
3. Keep exam names and disease names exactly as given.
4. Output plain text only, no markdown.
5. Write in Chinese (Simplified).

--- Global recommendations ---
{global_recs}

--- Per-disease recommendations ---
{per_disease_recs}
---

Write 2-3 natural sentences summarising the recommendations.
"""
)


def polish_exam_output(
    global_recs: list,
    per_d_recs:  list,
    llm:         ChatOpenAI,
) -> str:
    """
    Tool4 v2 output fields:
      global_recs:  [{exam_type, global_score, contributing_diseases}]
      per_d_recs:   [{disease, final_score, exams:[{exam_type, prob, ig}]}]
    """
    global_str = ", ".join(r["exam_type"] for r in global_recs) if global_recs else "（无）"
    per_d_str  = "; ".join(
        f"{rec['disease']}: {', '.join(e['exam_type'] for e in rec.get('exams', []))}"
        for rec in per_d_recs
    ) if per_d_recs else "（无）"

    try:
        return (POLISH_EXAM_PROMPT | llm).invoke({
            "global_recs":      global_str,
            "per_disease_recs": per_d_str,
        }).content.strip()
    except Exception as e:
        print(f"[polish] 检查润色失败：{e}")
        lines = []
        if global_recs:
            lines.append("总体建议您做：" + "、".join(r["exam_type"] for r in global_recs))
        for rec in per_d_recs:
            exams = "、".join(e["exam_type"] for e in rec.get("exams", []))
            if exams:
                lines.append(f"针对【{rec['disease']}】推荐：{exams}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 4. Exit condition
# ─────────────────────────────────────────────

def should_exit(tool3_result: dict) -> tuple[bool, str]:
    confidence = tool3_result.get("confidence_level", "low")
    top_status = tool3_result.get("top_status", "")
    if confidence == "high" and top_status in (CONFIRMED_STATUS, NO_DISEASE):
        return True, "诊断置信度已足够高，无需继续。"
    return False, ""


# ─────────────────────────────────────────────
# 5. User interaction
# ─────────────────────────────────────────────

def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"\n{prompt} (y/n): ").strip().lower()
        if ans in ("y", "yes", "是", "1"):
            return True
        if ans in ("n", "no", "否", "0"):
            return False
        print("请输入 y 或 n。")


def ask_disease_target(at_risk_diseases: list[dict]) -> str | None:
    if not at_risk_diseases:
        return None
    print("\n风险疾病列表：")
    for i, d in enumerate(at_risk_diseases, 1):
        print(f"  {i}. {d['disease']}")
    ans = input("如需针对特定疾病推荐检查，请输入编号（直接回车跳过）: ").strip()
    if not ans:
        return None
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(at_risk_diseases):
            return at_risk_diseases[idx]["disease"]
    except ValueError:
        pass
    return None


def ask_exam_results() -> tuple[str, list[str]]:
    print("\n请描述您的检查结果（例如：空腹血糖 8.5 mmol/L，HbA1c 7.2%）：")
    desc = input("> ").strip()
    if not desc:
        return "", []
    print("请输入已完成的检查类型名称，逗号分隔（例如：空腹血糖,HbA1c）：")
    done_raw  = input("> ").strip()
    done_list = [x.strip() for x in done_raw.split(",") if x.strip()] if done_raw else []
    return desc, done_list


# ─────────────────────────────────────────────
# 6. Input splitting
# ─────────────────────────────────────────────

SPLIT_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a medical information extraction assistant.
Split the following patient description into two parts:
1. symptom_description: only symptoms, feelings, and their durations.
2. exam_description: only examination types and their results/values.

Rules:
- If there are no examinations mentioned, set exam_description to null.
- Keep the original phrasing as much as possible.
- Output ONLY valid JSON, no markdown.

Input: {user_input}

Output:
{{
  "symptom_description": "<symptoms only>",
  "exam_description": "<exam results only or null>"
}}
"""
)


def split_input(user_input: str, llm: ChatOpenAI) -> tuple[str, str]:
    raw = (SPLIT_PROMPT | llm).invoke({"user_input": user_input}).content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        symp   = parsed.get("symptom_description", user_input) or user_input
        exam   = parsed.get("exam_description") or ""
        return symp.strip(), exam.strip()
    except Exception:
        return user_input, ""


# ─────────────────────────────────────────────
# 7. Core agent loop
# ─────────────────────────────────────────────

def run_agent_a(
    user_input:    str,
    llm:           ChatOpenAI,
    symptom_store: FAISS,
    exam_store:    FAISS,
    embeddings:    HuggingFaceEmbeddings,
    verbose:       bool = True,
) -> dict[str, Any]:
    """
    Main loop of Agent A.

    Args:
        user_input:    full patient description (symptoms and examination results, natural language)
        llm:           OpenAI LLM
        symptom_store: Tool1 FAISS index
        exam_store:    Tool2 FAISS index
        embeddings:    SapBERT embeddings
        verbose:       whether to print intermediate logs

    Returns:
        final session_state
    """
    # -- Split the input -----------------------------------------------
    if verbose:
        print("[Agent] 拆分用户输入（症状 / 检查）...")
    symptom_input, exam_input = split_input(user_input, llm)
    if verbose:
        print(f"  症状：{symptom_input}")
        print(f"  检查：{exam_input or '（无）'}")

    # ── session state ────────────────────────────────────────
    session_state = {
        "user_input":    user_input,
        "symptom_input": symptom_input,
        "exam_input":    exam_input,
        "done_exams":    [],
        "turn":          0,
        "tool1_result":  None,
        "tool3_result":  None,
        "history":       [],
    }

    print("\n" + "="*60)
    print("Agent A 开始（意图A：开放式诊断）")
    print("="*60)

    while True:
        turn = session_state["turn"]
        print(f"\n─── 第 {turn + 1} 轮 ───────────────────────────────────────")

        # -- Tool1: symptom matching (first turn only; symptoms do not change) --
        if turn == 0 or session_state["tool1_result"] is None:
            if verbose:
                print("[Agent] 运行 Tool1（症状匹配）...")
            t1_raw = symptom_matching_pipeline(
                session_state["symptom_input"], llm, symptom_store, embeddings
            )
            session_state["tool1_result"] = t1_convert(t1_raw)

        t1_out = session_state["tool1_result"]

        # -- Tool2: examination matching (rerun every turn; exam_input may change) --
        current_exam_input = session_state["exam_input"]
        if current_exam_input:
            if verbose:
                print("[Agent] 运行 Tool2（检查匹配）...")
            t2_raw    = exam_matching_pipeline(
                current_exam_input, llm, exam_store, embeddings
            )
            t2_result = t2_convert(t2_raw)
        else:
            t2_raw    = None
            t2_result = {"candidate_diseases": []}

        # -- Tool3: combined diagnosis -------------------------------------
        if verbose:
            print("[Agent] 运行 Tool3（综合诊断）...")
        t3_result = combine_results(t1_out, t2_result)
        t3_result = t3_convert(t3_result)

        # Pass matched_symptoms / matched_exams through at the agent level (in case Tool3 drops them)
        t1_map = {d["disease"]: d for d in t1_out.get("candidate_diseases", [])}
        t2_map = {d["disease"]: d for d in t2_result.get("candidate_diseases", [])}
        for d in t3_result.get("candidate_diseases", []):
            if not d.get("matched_symptoms") and d["disease"] in t1_map:
                d["matched_symptoms"] = t1_map[d["disease"]].get("matched_symptoms", [])
            if not d.get("matched_exams") and d["disease"] in t2_map:
                d["matched_exams"] = t2_map[d["disease"]].get("matched_exams", [])

        session_state["tool3_result"] = t3_result

        # -- Output tiers + LLM rewriting ---------------------------------
        candidate_diseases = t3_result.get("candidate_diseases", [])
        classified         = classify_output(candidate_diseases)
        output_msg         = format_output_message(classified)

        if verbose:
            print("[Agent] LLM润色输出...")
        polished_msg = polish_output(
            output_msg, t3_result, llm,
            t1_raw=t1_out,
            t2_raw=t2_result,
        )

        print(f"\n[诊断结果]\n{polished_msg}")
        session_state["history"].append({
            "turn":            turn + 1,
            "output_raw":      output_msg,
            "output_polished": polished_msg,
            "t3_result":       t3_result,
        })

        # -- Exit check ---------------------------------------------------
        exit_flag, exit_reason = should_exit(t3_result)
        if exit_flag:
            print(f"\n[Agent] 会话结束：{exit_reason}")
            break

        if not ask_yes_no("是否需要进一步帮您判断风险？"):
            print("[Agent] 用户选择结束，会话结束。")
            break

        # -- Tool4: examination recommendation -----------------------------
        at_risk        = classified["at_risk"]
        all_candidates = candidate_diseases
        target_disease = ask_disease_target(at_risk) if at_risk else None

        if target_disease:
            tool4_input = [d for d in all_candidates if d["disease"] == target_disease]
            if not tool4_input:
                tool4_input = all_candidates
        else:
            tool4_input = all_candidates

        if verbose:
            print("[Agent] 运行 Tool4（推荐检查）...")
        t4_result    = recommend_exams(tool4_input, session_state["done_exams"])
        t4_result    = t4_convert(t4_result)

        global_recs = t4_result.get("global_recommendations", [])
        per_d_recs  = t4_result.get("per_disease_recommendations", [])

        if verbose:
            print("[Agent] LLM润色检查推荐...")
        polished_exam = polish_exam_output(global_recs, per_d_recs, llm)
        print(f"\n[推荐检查]\n{polished_exam}")

        # -- Intent C: add examination results ---------------------------------
        if not ask_yes_no("是否补充检查结果？"):
            print("[Agent] 用户选择不补充，会话结束。")
            break

        new_exam_desc, new_done = ask_exam_results()
        if new_exam_desc:
            if session_state["exam_input"]:
                session_state["exam_input"] += "。" + new_exam_desc
            else:
                session_state["exam_input"] = new_exam_desc
        session_state["done_exams"].extend(new_done)
        session_state["turn"] += 1

    print("\n" + "="*60)
    print("Agent A 会话结束")
    print("="*60)
    return session_state


# ─────────────────────────────────────────────
# 8. Initialisation & entry point
# ─────────────────────────────────────────────

def init_agent(
    openai_api_key:     str,
    symptom_index_path: str  = "symptom_faiss_index",
    exam_index_path:    str  = "exam_faiss_index",
    rebuild_index:      bool = False,
) -> tuple[ChatOpenAI, FAISS, FAISS, HuggingFaceEmbeddings]:
    """
    Initialise the LLM, vector indexes and embeddings.

    Args:
        rebuild_index: True  -> rebuild the FAISS index (first run or after the KG is updated)
                       False -> load the existing index (normal run)
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

    print("[init] Agent A 初始化完成。")
    return llm, symp_store, exam_store, embeddings