# src/adk_search_sota.py
import os, ast, json, re
from typing import Optional, List
from google.genai import types
from google.adk import agents
from google.adk.agents import callback_context as cbx
from google.adk.models import llm_response as llm_resp
from google.adk.models import llm_request as llm_req
from google.adk.tools.google_search_tool import google_search

# ==== PROMPT (VERBATIM from MLE-STAR) ====
MODEL_RETRIEVAL_INSTR = """# Competition
{task_summary}

# Your task
- List {num_model_candidates} state-of-the-art effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]

Output rules:
- Output ONLY the JSON array (no explanations, no markdown).
- Do NOT wrap the output in any code fences (e.g., ```json, ```python, or ```).
- Ensure each item has runnable Python in 'example_code'.
- If uncertain, return a single best candidate.
Example shape: [{{"model_name":"<name>","example_code":"<python>"}}]
"""

def get_model_retriever_agent_instruction(context: cbx.ReadonlyContext) -> str:
    """Formats the retrieval prompt with state-backed variables, with optional retry hint."""
    task_summary = context.state.get("task_summary", "")
    num_model_candidates = context.state.get("num_model_candidates", 1)
    base = MODEL_RETRIEVAL_INSTR.format(
        task_summary=task_summary,
        num_model_candidates=num_model_candidates,
    )
    # Optional: inject guideline hints from pipeline guideline
    guideline_hint = context.state.get("guideline_hint")
    if guideline_hint:
        base += (
            "\n# Guideline hints\n"
            "Use these preferences from the pipeline guideline when selecting models (favor alignment, not strict enforcement):\n"
            f"{guideline_hint}\n"
        )
    hint = context.state.get("regen_hint_msg")
    return base + (("\n\n" + hint) if hint else "")

def _get_text_from_response(resp: llm_resp.LlmResponse) -> str:
    """Concatenate text parts from the LLM response."""
    txt = ""
    if resp.content and resp.content.parts:
        for p in resp.content.parts:
            if hasattr(p, "text"):
                txt += p.text or ""
    return txt

def _strip_fences(s: str) -> str:
    """Remove common markdown code fences like ```json, ```python, ``` and even multiple backticks.
    Also removes an optional trailing newline immediately after the fence marker.
    """
    # Remove opening fences like ```json\n or ````python\n
    s = re.sub(r"```+\w*\n", "", s)
    # Remove any remaining triple-or-more backticks
    s = re.sub(r"```+", "", s)
    return s

def _extract_json_arrays(s: str):
    """
    Find candidate substrings that look like JSON arrays of objects: [ {...}, {...}, ... ]
    Non-greedy and dotall to handle newlines.
    """
    pattern = r"\[\s*(?:\{.*?\})\s*(?:,\s*\{.*?\}\s*)*\s*\]"
    return re.findall(pattern, s, flags=re.S)

def _looks_like_code(s: str) -> bool:
    if not s:
        return False
    t = _strip_fences(s).strip()
    # Heuristic signals of Python code
    has_keyword = any(k in t for k in [
        "import ", "from ", "def ", "class ", " = ", "fit(", "predict(", "torch", "sklearn", "pandas", "numpy"
    ])
    long_enough = len(t) >= 30 and ("\n" in t or ";" in t or "(" in t)
    not_pure_prose = not re.search(r"\b(refer to|paper|implementation details|\[\d+\])\b", t, flags=re.I)
    return has_keyword and long_enough and not_pure_prose

def get_model_candidates(
    callback_context: cbx.CallbackContext,
    llm_response: llm_resp.LlmResponse,
) -> Optional[llm_resp.LlmResponse]:
    """Parse JSON list; only finish when example_code looks like runnable Python."""
    # Rebuild text from response parts
    text = ""
    if llm_response.content and llm_response.content.parts:
        for p in llm_response.content.parts:
            if getattr(p, "text", None):
                text += p.text
    if not text:
        return None

    raw = _strip_fences(text)
    candidates = _extract_json_arrays(raw)

    items = None
    # try each candidate
    for cand in candidates:
        parsed = None
        try:
            parsed = json.loads(cand)
        except Exception:
            try:
                parsed = ast.literal_eval(cand)
            except Exception:
                parsed = None
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            filtered = [it for it in parsed if _looks_like_code(it.get("example_code", ""))]
            if filtered:
                items = filtered
                break

    # fallback: smallest bracket region
    if items is None:
        opens = [m.start() for m in re.finditer(r"\[", raw)]
        closes = [m.start() for m in re.finditer(r"\]", raw)]
        for i in opens:
            for j in closes:
                if j > i and (j - i) <= 20000:
                    cand = raw[i:j+1]
                    parsed = None
                    try:
                        parsed = json.loads(cand)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(cand)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        filtered = [it for it in parsed if _looks_like_code(it.get("example_code", ""))]
                        if filtered:
                            items = filtered
                            break
            if items is not None:
                break

    if items is None:
        # escalate hints with retry counter; do not raise, let LoopAgent retry
        retry = int(callback_context.state.get("retry_count", 0) or 0) + 1
        callback_context.state["retry_count"] = retry
        if retry == 1:
            hint = (
                "Output ONLY a JSON array following the schema; no prose, no code fences. "
                "Ensure each 'example_code' is runnable Python."
            )
        elif retry == 2:
            hint = (
                "Return pure JSON array like: [{\"model_name\":\"BERT\",\"example_code\":\"import ...\"}] — no markdown, no text, no fences."
            )
        else:
            hint = (
                "Return a SINGLE best item as a JSON array with runnable Python under 'example_code' (no code fences)."
            )
        callback_context.state["regen_hint_msg"] = hint
        return None

    callback_context.state["retrieved_models"] = items
    callback_context.state["init_model_finish"] = True
    # clear regen hint safely for ADK State (not a plain dict)
    try:
        del callback_context.state["regen_hint_msg"]
    except Exception:
        callback_context.state["regen_hint_msg"] = None
    return None

def check_model_finish(
    callback_context: cbx.CallbackContext,
    llm_request: llm_req.LlmRequest,
) -> Optional[llm_resp.LlmResponse]:
    """If parsing completed in a previous iteration, stop the loop."""
    if callback_context.state.get("init_model_finish", False):
        return llm_resp.LlmResponse()
    return None

def make_search_sota_root_agent(task_summary: str, k: int = 1, guideline: Optional[object] = None):
    """
    Build a root agent that:
      - bootstraps task_summary & k into the shared state
      - runs a LoopAgent with the retriever agent until parsing succeeds
    - emits a single-element JSON array with the first result; raises an error if no valid result
    """
    def bootstrap_state(callback_context: cbx.CallbackContext) -> Optional[types.Content]:
        callback_context.state["task_summary"] = task_summary
        callback_context.state["num_model_candidates"] = k
        # Stash compact guideline hints if provided
        if guideline is not None:
            try:
                if isinstance(guideline, (dict, list)):
                    txt = json.dumps(guideline, ensure_ascii=False)
                else:
                    txt = str(guideline)
                # Keep it short to avoid overwhelming the prompt
                callback_context.state["guideline_hint"] = (txt[:4000] + "…") if len(txt) > 4000 else txt
            except Exception:
                callback_context.state["guideline_hint"] = str(guideline)
        return None

    def emit_result(callback_context: cbx.CallbackContext) -> Optional[types.Content]:
        items = callback_context.state.get("retrieved_models", [])
        out = []
        if items:
            first = items[0]
            out = [{
                "model_name": str(first.get("model_name", "")).strip(),
                "example_code": str(first.get("example_code", "")).strip()
            }]
        else:
            # No valid candidates — signal hard failure
            raise RuntimeError("SOTA search failed: no valid model candidates retrieved.")
        return types.Content(parts=[types.Part(text=json.dumps(out, ensure_ascii=False))])

    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")

    model_retriever_agent = agents.Agent(
        model=model_name,
        name="model_retriever_agent_1",
        description="Retrieve recent effective models + example code.",
        instruction=get_model_retriever_agent_instruction,
        tools=[google_search],  
        before_model_callback=check_model_finish,
        after_model_callback=get_model_candidates,
        generate_content_config=types.GenerateContentConfig(temperature=0.7),
        include_contents="none",
    )

    model_retriever_loop = agents.LoopAgent(
        name="model_retriever_loop",
        description="Retrieve models until parse succeeds.",
        sub_agents=[model_retriever_agent],
        max_iterations=10,
    )

    root = agents.SequentialAgent(
        name="sota_search_root",
        description="Root agent for Search SOTA",
        sub_agents=[model_retriever_loop],
        before_agent_callback=bootstrap_state,
        after_agent_callback=emit_result,   
    )
    return root