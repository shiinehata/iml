from .base_prompt import BasePrompt

class CandidateSelectorPrompt(BasePrompt):
    """
    A prompt to guide the LLM in selecting the best candidate code from a list of options.
    """

    def default_template(self) -> str:
        """Default prompt template"""
        return """
You are an expert Python programmer and data scientist. Your task is to analyze several candidate scripts and select the one that is most likely to achieve the best performance for the given problem.

**Context:**
- **Problem Description:** {description_analysis}
- **Data Profiling Overview:** {profiling_result}
- **High-Level Plan (Guideline):** {guideline}
- **Original Code's Feedback:** {feedback}

**Candidate Scripts:**
{candidate_scripts}

**Instructions:**
1.  Review each candidate script carefully.
2.  Compare them against each other and the original feedback.
3.  Evaluate them based on potential for high performance, robustness, and adherence to the improvement suggestions.
4.  Your final output must be only the full, complete code of the single best candidate.
5.  **Do not add any explanation, commentary, or formatting.** Your output should be only the raw Python code for the chosen script, ready to be executed.

**Example of a valid output:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ... rest of the chosen candidate's code
```
"""

    def __init__(self, manager, llm_config, **kwargs):
        super().__init__(manager, llm_config, **kwargs)

    def build(self) -> str:
        candidate_scripts = ""
        for i, code in enumerate(self.manager.candidates):
            candidate_scripts += f"### CANDIDATE {i+1} ###\n```python\n{code}\n```\n\n"

        return self.template.format(
            description_analysis=self.manager.description_analysis,
            profiling_result=self.manager.profiling_result,
            guideline=self.manager.guideline,
            feedback=self.manager.feedback,
            candidate_scripts=candidate_scripts.strip(),
        )

    def parse(self, response: str) -> any:
        """Parse the LLM response"""
        return response
