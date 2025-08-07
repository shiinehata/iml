import logging
import re
from .base_agent import BaseAgent
from ..prompts.candidate_generator_prompt import CandidateGeneratorPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class CandidateGeneratorAgent(BaseAgent):
    """
    An agent that generates multiple candidate code versions based on feedback.
    It ensures that each generated candidate is executable.
    """
    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="candidate_generator",
            multi_turn=self.llm_config.multi_turn,
        )


    def __call__(self) -> dict:
        """
        Generates and validates candidate code versions sequentially.
        """
        self.manager.log_agent_start("Generating candidate codes sequentially based on feedback...")

        if not hasattr(self.manager, "feedback"):
            error_msg = "Missing feedback in manager for candidate generation."
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        prompt_handler = CandidateGeneratorPrompt(
            manager=self.manager,
            llm_config=self.llm_config,
        )

        valid_candidates = []
        max_candidates = 3
        max_retries_per_candidate = 3

        for i in range(max_candidates):
            logger.info(f"--- Generating Candidate {i+1}/{max_candidates} ---")
            found_valid_candidate_for_slot = False
            for attempt in range(max_retries_per_candidate):
                logger.info(f"Attempt {attempt + 1}/{max_retries_per_candidate} for Candidate {i+1}...")
                
                try:
                    prompt = prompt_handler.build(
                        candidate_number=i + 1,
                        previous_candidates=valid_candidates
                    )
                    response = self.llm.assistant_chat(prompt)
                    code = prompt_handler.parse(response)

                    if not code:
                        logger.warning("LLM returned empty code. Retrying...")
                        continue

                    logger.info(f"Validating generated code for Candidate {i+1}...")
                    execution_result = self.manager.execute_code(code)

                    if execution_result["success"]:
                        logger.info(f"Candidate {i+1} is valid and has been added.")
                        valid_candidates.append(code)
                        found_valid_candidate_for_slot = True
                        break  # Exit retry loop and move to the next candidate
                    else:
                        logger.warning(f"Candidate {i+1} (attempt {attempt+1}) failed validation. Retrying. Error: {execution_result['stderr']}")

                except Exception as e:
                    logger.error(f"An error occurred during generation for Candidate {i+1} (attempt {attempt + 1}): {e}")
            
            if not found_valid_candidate_for_slot:
                logger.error(f"Could not generate a valid Candidate {i+1} after {max_retries_per_candidate} attempts. Moving on.")

        if valid_candidates:
            self.manager.log_agent_end(f"Successfully generated {len(valid_candidates)} valid candidate codes.")
            return {"status": "success", "candidates": valid_candidates}
        else:
            error_msg = "Failed to generate any valid candidate codes after all attempts."
            self.manager.log_agent_end(error_msg)
            return {"status": "failed", "error": error_msg}

    def _parse_candidates(self, response: str) -> list[str]:
        """
        Parses the LLM response to extract individual code candidates.
        """
        # The pattern looks for ### CANDIDATE X ### and captures the code block that follows.
        # re.DOTALL allows `.` to match newlines.
        pattern = r"### CANDIDATE \d+ ###\s*```python\n(.*?)\n```"
        candidates = re.findall(pattern, response, re.DOTALL)
        return [c.strip() for c in candidates]
