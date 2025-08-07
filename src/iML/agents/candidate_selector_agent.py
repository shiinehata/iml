import logging
import re
from .base_agent import BaseAgent
from ..prompts.candidate_selector_prompt import CandidateSelectorPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class CandidateSelectorAgent(BaseAgent):
    """
    An agent that uses an LLM to select the best code candidate from a list.
    """
    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="candidate_selector",
            multi_turn=self.llm_config.multi_turn,
        )


    def __call__(self) -> dict:
        """
        Selects the best candidate code.

        Returns:
            A dictionary containing the selected code or an error message.
        """
        self.manager.log_agent_start("Selecting the best candidate code...")

        if not hasattr(self.manager, "candidates") or not self.manager.candidates:
            error_msg = "No candidates available for selection."
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        prompt_handler = CandidateSelectorPrompt(
            manager=self.manager,
            llm_config=self.llm_config,
        )

        try:
            prompt = prompt_handler.build()
            response = self.llm.assistant_chat(prompt)
            parsed_response = prompt_handler.parse(response)
            # The LLM is instructed to return only the code, so we do a simple cleanup
            selected_code = self._extract_code(parsed_response)
            self.manager.save_and_log_states(selected_code, "final_selected_code.py")

            if not selected_code:
                raise ValueError("LLM did not return a valid code block.")

            self.manager.log_agent_end("Best candidate selected successfully.")
            return {"status": "success", "selected_code": selected_code}
        except Exception as e:
            error_msg = f"An error occurred while selecting the best candidate: {e}"
            logger.error(error_msg)
            self.manager.log_agent_end(error_msg, level="error")
            return {"status": "failed", "error": error_msg}

    def _extract_code(self, response: str) -> str:
        """
        Extracts the Python code from the LLM's response.
        Handles cases where the code is wrapped in markdown.
        """
        # Pattern to find a python code block
        match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no markdown block is found, assume the whole response is the code
        return response.strip()
