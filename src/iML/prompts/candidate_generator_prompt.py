from .base_prompt import BasePrompt
import re

class CandidateGeneratorPrompt(BasePrompt):
    """
    A prompt to guide the LLM in generating multiple candidate code versions based on feedback.
    """

    def default_template(self) -> str:
        """Default prompt template"""
        return """
You are an expert Python programmer and data scientist. Your task is to generate ONE improved version of the original Python script based on the provided feedback.

**Context:**
- **Problem Description:** {description_analysis}
- **Data Profiling Overview:** {profiling_result}
- **High-Level Plan (Guideline):** {guideline}
- **Original Code:**
  ```python
  {assembled_code}
  ```
- **Feedback for Improvement:**
  {feedback}
- **Previously Generated Valid Candidates:**
  {previous_candidates}

**Instructions:**
1.  Carefully read the feedback and understand the suggested improvements.
2.  Review the previously generated candidates to ensure your new suggestion is **distinct** and explores a **different improvement strategy**.
3.  Generate **one new candidate script** (Candidate #{candidate_number}). This script should incorporate one or more of the suggestions from the feedback.
4.  **Crucially, the candidate must be a complete and runnable Python script.** Do not use placeholders.
5.  **IMPORTANT**: Do not use command-line arguments (e.g., `sys.argv`). The script must run without any external arguments.
6.  **CRITICAL**: The original code contains absolute paths to data files. You **MUST** preserve these exact paths in your generated code. Do not change them.
7.  Format your output as a single, raw Python code block, enclosed in ```python ... ```.

**Example Output Format:**
```python
# Complete Python code for the new candidate...
import pandas as pd
# ... rest of the script
```
"""

    def __init__(self, manager, llm_config, **kwargs):
        super().__init__(manager, llm_config, **kwargs)

    def build(self, candidate_number: int, previous_candidates: list[str]) -> str:
        
        previous_candidates_str = "No valid candidates have been generated yet."
        if previous_candidates:
            previous_candidates_str = ""
            for i, code in enumerate(previous_candidates):
                previous_candidates_str += f"### PREVIOUS CANDIDATE {i+1} ###\n```python\n{code}\n```\n\n"

        return self.template.format(
            description_analysis=self.manager.description_analysis,
            profiling_result=self.manager.profiling_result,
            guideline=self.manager.guideline,
            assembled_code=self.manager.assembled_code,
            feedback=self.manager.feedback,
            candidate_number=candidate_number,
            previous_candidates=previous_candidates_str.strip(),
        )

    def parse(self, response: str) -> any:
        """Parse the LLM response to extract the code block."""
        match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback if the LLM doesn't use markdown
        return response.strip()
