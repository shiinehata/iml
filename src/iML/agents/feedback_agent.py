import logging
from .base_agent import BaseAgent
from ..prompts.feedback_prompt import FeedbackPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class FeedbackAgent(BaseAgent):
    """
    An agent that analyzes the successfully executed code and provides feedback for improvement.
    """
    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="feedback",
            multi_turn=self.llm_config.multi_turn,
        )


    def __call__(self) -> dict:
        """
        Analyzes the code and returns feedback.

        Returns:
            A dictionary containing the feedback or an error message.
        """
        self.manager.log_agent_start("Analyzing code for feedback...")

        # Ensure the required information is available in the manager
        if not hasattr(self.manager, "description_analysis") or \
           not hasattr(self.manager, "profiling_result") or \
           not hasattr(self.manager, "guideline") or \
           not hasattr(self.manager, "assembled_code"):
            error_msg = "Missing necessary data in manager for feedback generation."
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        prompt_handler = FeedbackPrompt(
            manager=self.manager,
            llm_config=self.llm_config,
        )

        try:
            prompt = prompt_handler.build()
            response = self.llm.assistant_chat(prompt)
            parsed_response = prompt_handler.parse(response)
            self.manager.log_agent_end("Feedback generated successfully.")
            return {"status": "success", "feedback": parsed_response}
        except Exception as e:
            error_msg = f"An error occurred while generating feedback: {e}"
            logger.error(error_msg)
            self.manager.log_agent_end(f"Feedback generation failed: {error_msg}", level="error")
            return {"status": "failed", "error": error_msg}
