from .base_prompt import BasePrompt

class FeedbackPrompt(BasePrompt):
    """
    A prompt to guide the LLM in analyzing the generated code and providing feedback for improvement.
    """

    def default_template(self) -> str:
        """Default prompt template"""
        return """
You are an expert Python programmer and data scientist. Your task is to analyze the provided Python script, which has been successfully executed, and provide constructive feedback on how to improve its performance and robustness.

**Context:**
- **Problem Description:** {description_analysis}
- **Data Profiling Overview:** {profiling_result}
- **High-Level Plan (Guideline):** {guideline}

**Code to Analyze:**
```python
{assembled_code}
```

**Instructions:**
Please analyze the code and provide feedback in the following format:
1.  **Strengths:** What are the good aspects of this code? (e.g., clarity, correctness, good use of libraries).
2.  **Weaknesses:** What are the potential issues or areas for improvement? (e.g., performance bottlenecks, inefficient algorithms, lack of error handling, hardcoded values that could be generalized).
3.  **Suggestions for Improvement:** Provide specific, actionable suggestions to address the weaknesses. Focus on changes that could lead to better model performance (e.g., different feature engineering techniques, alternative models, hyperparameter tuning strategies).

Your feedback should be clear, concise, and directly aimed at improving the final model's predictive accuracy or efficiency.
"""

    def __init__(self, manager, llm_config, **kwargs):
        super().__init__(manager, llm_config, **kwargs)

    def build(self) -> str:
        return self.template.format(
            description_analysis=self.manager.description_analysis,
            profiling_result=self.manager.profiling_result,
            guideline=self.manager.guideline,
            assembled_code=self.manager.assembled_code,
        )

    def parse(self, response: str) -> any:
        """Parse the LLM response"""
        return response
