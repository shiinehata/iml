# src/iML/prompts/assembler_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

class AssemblerPrompt(BasePrompt):
    """
    Prompt handler to assemble and fix final code.
    """

    def default_template(self) -> str:
        """Default template to request LLM to rewrite and fix code."""
        return """
You are a senior ML engineer finalizing a project. You have been given a Python script that combines preprocessing and modeling.
Your task is to ensure the script is clean, robust, and correct.

## REQUIREMENTS:
1.  **Final Script**: The output must be a single, standalone, executable Python file and it should be run on the real data.
2.  **Validation Score**: If validation data is available, you MUST calculate and print a relevant validation score.
3.  **Absolute Output Path**: The script MUST save `submission.csv` to the following absolute path: `{output_path}`.
4.  **Error Handling**: Maintain the `try...except` block for robust execution.
5.  **Clarity**: Ensure the final script is clean and well-structured.
6.  **Data Loading**: Keep the data loading code of the preprocessing code. DO NOT CHANGE THE FILE PATHS FROM THE ORIGINAL CODE.
7.  **Do not add any other code.**

## ORIGINAL CODE:
```python
{original_code}
```
{retry_context}
## INSTRUCTIONS:
Based on the context above, generate the complete and corrected Python code. The output should be ONLY the final Python code.

## FINAL, CORRECTED CODE:
"""

    def build(self, original_code: str, output_path: str, error_message: str = None) -> str:
        """Build prompt to assemble or fix code."""
        
        retry_context = ""
        if error_message:
            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The code above failed with the following error.

### Error Message:
```
{error_message}
```

### FIX INSTRUCTIONS:
1.  Analyze the error message and the original code carefully.
2.  Fix the specific issue that caused the error.
3.  Generate a new, complete, and corrected version of the Python code that resolves the issue and meets all requirements.
"""
        
        prompt = self.template.format(
            original_code=original_code,
            output_path=output_path,
            retry_context=retry_context
        )
        
        self.manager.save_and_log_states(prompt, "assembler_prompt.txt", per_iteration=True)
        return prompt

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "final_assembled_code.py", per_iteration=True)
        return code
