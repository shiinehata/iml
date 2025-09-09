# src/iML/agents/guideline_agent.py
import logging
import json
from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..prompts.guideline_prompt import GuidelinePrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class GuidelineAgent(BaseAgent):
    """
    This agent creates a detailed guideline to solve the problem,
    based on description information and data profiling results.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        
        self.llm_config = llm_config
        self.prompt_template = prompt_template

        self.prompt_handler = GuidelinePrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )

        # Initialize LLM
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="guideline_agent",
            multi_turn=self.llm_config.get('multi_turn', False)
        )

    def __call__(self) -> Dict[str, Any]:
        """
        Execute agent to create guideline.
        """
        self.manager.log_agent_start("GuidelineAgent: Starting guideline generation...")

        description_analysis = self.manager.description_analysis
        # Prefer summarized profiling to avoid noise; fallback to raw if missing
        profiling_result = getattr(self.manager, "profiling_summary", None)
        if not profiling_result:
            profiling_result = getattr(self.manager, "profiling_result", None)

        if not description_analysis or "error" in description_analysis:
            logger.error("GuidelineAgent: description_analysis is missing.")
            return {"error": "description_analysis not available."}
        
        if not profiling_result or "error" in profiling_result:
            logger.error("GuidelineAgent: profiling summary/result is missing.")
            return {"error": "profiling_result not available."}

        # Build prompt with SOTA retrieval results if present
        retrieved_models = getattr(self.manager, "retrieved_models", None)
        prompt = self.prompt_handler.build(
            description_analysis=description_analysis,
            profiling_result=profiling_result,
            retrieved_models=retrieved_models,
        )

        # Call LLM
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "guideline_raw_response.txt")

        # Analyze results
        guideline = self.prompt_handler.parse(response)

        # Enforce usage of SOTA model(s) when available
        def _names_from_sota(items: List[dict]) -> List[str]:
            return [str(x.get("model_name", "")).strip() for x in (items or []) if isinstance(x, dict)]

        def _uses_any_sota(g: Dict[str, Any], allowed: List[str]) -> bool:
            try:
                sel = g.get("modeling", {}).get("model_selection", [])
                if not isinstance(sel, list):
                    return False
                sel_lower = {str(s).strip().lower() for s in sel}
                for name in allowed:
                    if name and name.lower() in sel_lower:
                        return True
                return False
            except Exception:
                return False

        if retrieved_models:
            allowed_names = _names_from_sota(retrieved_models)
            def _ensure_sota_model_block(g: Dict[str, Any], chosen: str, code: str):
                g.setdefault("modeling", {})
                g["modeling"]["sota_model"] = {
                    "model_name": chosen,
                    "example_code": code,
                }

            if allowed_names and not _uses_any_sota(guideline, allowed_names):
                # Ask LLM to revise JSON to include one of the allowed SOTA models
                try:
                    revision_prompt = (
                        "Revise the following JSON to include EXACTLY ONE of these model names in 'modeling.model_selection' "
                        "(as a JSON list of strings). Also add 'modeling.sota_model' with both 'model_name' and 'example_code' "
                        "(the example_code must be runnable Python). Keep other content, and output JSON only.\n\n"
                        f"Allowed model names: {json.dumps(allowed_names, ensure_ascii=False)}\n\n"
                        f"Current JSON:```json\n{json.dumps(guideline, ensure_ascii=False)}\n```"
                    )
                    self.manager.save_and_log_states(revision_prompt, "guideline_revision_prompt.txt")
                    rev_resp = self.llm.assistant_chat(revision_prompt)
                    self.manager.save_and_log_states(rev_resp, "guideline_revision_raw_response.txt")
                    try:
                        revised = json.loads(rev_resp.strip().replace("```json", "").replace("```", ""))
                    except Exception:
                        revised = guideline
                    # Validate presence of chosen model and sota_model block
                    valid = _uses_any_sota(revised, allowed_names)
                    try:
                        sm = revised.get("modeling", {}).get("sota_model", {})
                        valid = valid and isinstance(sm, dict) and bool(sm.get("model_name")) and bool(sm.get("example_code"))
                    except Exception:
                        valid = False
                    if valid:
                        guideline = revised
                        self.manager.save_and_log_states(
                            json.dumps(guideline, indent=2, ensure_ascii=False),
                            "guideline_revision_response.json",
                        )
                except Exception:
                    pass

                # Final fallback: programmatically inject the first SOTA model if still missing
                if not _uses_any_sota(guideline, allowed_names):
                    try:
                        # choose the first retrieved item to inject
                        first_item = retrieved_models[0] if isinstance(retrieved_models, list) and retrieved_models else {}
                        first = (first_item.get("model_name") or allowed_names[0])
                        code = first_item.get("example_code") or ""
                        guideline.setdefault("modeling", {})
                        ms = guideline["modeling"].get("model_selection")
                        if not isinstance(ms, list):
                            guideline["modeling"]["model_selection"] = [first]
                        else:
                            guideline["modeling"]["model_selection"].append(first)
                        _ensure_sota_model_block(guideline, first, code)
                        self.manager.save_and_log_states(
                            json.dumps(guideline, indent=2, ensure_ascii=False),
                            "guideline_response_forced.json",
                        )
                    except Exception:
                        pass

        # Persist the final guideline (post-enforcement) to the canonical artifact
        try:
            self.manager.save_and_log_states(
                json.dumps(guideline, indent=2, ensure_ascii=False),
                "guideline_response.json",
            )
        except Exception:
            pass

        self.manager.log_agent_end("GuidelineAgent: Guideline generation COMPLETED.")
        return guideline
