# src/iML/prompts/preprocessing_coder_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

class PreprocessingCoderPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for data preprocessing.
    """

    def default_template(self) -> str:
        """Default template to request LLM to generate code."""
        return """
You are a professional Machine Learning Engineer.
Generate complete and executable Python preprocessing code for the dataset below.
IMPORTANT: Preprocess data by batch using generators to reduce memory usage.
IMPORTANT: DO NOT CREATE DUMMY DATA.
## DATASET INFO:
- Name: {dataset_name}
- Task: {task_desc}
- Input: {input_desc}
- Output: {output_desc}
- Data files: {data_file_desc}
- File paths: {file_paths} (LOAD DATA FROM THESE PATHS)

## PREPROCESSING GUIDELINES:
{preprocessing_guideline}

## TARGET INFO:
{target_info}

## RELEVANT TUTORIALS:
{tutorials_section}

## REQUIREMENTS:
1. **RECOMMENDED: Consider using the tutorial code examples as a reference when they are helpful.** You may adapt code patterns, imports, and methodologies from the tutorial code blocks if they suit your needs.
2. Generate COMPLETE, EXECUTABLE Python code using an appropriate approach for your data type.
3. Handle file loading exactly as the provided paths, DO NOT CREATE DUMMY DATA FILES.
4. Create a function `preprocess_data()` that takes a dictionary of file paths and returns preprocessed data in an appropriate format (could be arrays, generators, datasets, etc.).
5. **SUGGESTED: Consider tutorial data loading and preprocessing patterns when they match your task, but adapt as needed.**
6. Include basic error handling and data validation within the function.
7. Limit comments in the code.
8. Preprocess both the train and test data consistently.
9. IMPORTANT: The main execution block (`if __name__ == "__main__":`) should test the function with the actual file paths.
10. **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error and then **exit with a non-zero status code** using `sys.exit(1)`.
11. DO NOT USE NLTK even if the tutorial uses it.
12. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
13. The provided file paths are the only valid paths to load the data. Do not create any dummy data files.
14. **OPTIONAL: When tutorials provide helpful code examples, you may use those patterns and structures as inspiration, but feel free to adapt or use different approaches if they better fit your specific needs.**

## CODE STRUCTURE:
```python
# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

def preprocess_data(file_paths: dict):
    \"\"\"
    Preprocess data according to guidelines.
    Preprocesses data and returns batch generators.
    Args:
        file_paths: A dictionary of file paths for data splits.
    Returns:
        A tuple of generators, one for each data split (e.g., (train_gen, val_gen, test_gen)).
    \"\"\"
    # Your preprocessing code here
    
    # Placeholder return
    train_generator, val_generator, test_generator = (None, None, None)
    
    return train_generator, val_generator, test_generator

# Test the function
if __name__ == "__main__":
    try:
        # This assumes the script is run from a directory where it can access the paths
        file_paths = {file_paths_main}
        train_gen, val_gen, test_gen = preprocess_data(file_paths)
        print("Generators initialized.")

        
        print("\\nPreprocessing script and generator test executed successfully!")

    except Exception as e:
        print(f"An error occurred during preprocessing test: {{e}}", file=sys.stderr)
        sys.exit(1)
````
"""

    def build(self, guideline: Dict, description: Dict, previous_code: str = None, error_message: str = None, tutorials_content: str = "") -> str:
        """Build prompt to generate preprocessing code."""
        
        preprocessing_guideline = guideline.get('preprocessing', {})
        target_info = guideline.get("target_identification", {})
        
        # Format tutorials section
        if tutorials_content.strip():
            tutorials_section = f"""The following tutorials provide code examples and approaches that you may find helpful for your task:

{tutorials_content}

**TUTORIAL GUIDANCE (OPTIONAL - USE IF HELPFUL):**
1. **Consider reviewing** the tutorial code examples above for potential preprocessing approaches
2. **You may adapt** import patterns, data loading utilities, and preprocessing methods shown if they suit your needs
3. **Tutorial coding structures** can serve as inspiration for your implementation
4. **Feel free to use** the libraries and methods from tutorials when they fit your requirements
5. **Other approaches are acceptable** - use your judgment on what works best for your specific dataset
6. **Tutorial methods can be adapted** to your specific dataset and requirements as needed"""
        else:
            tutorials_section = "No specific tutorials found for this task type. Use TensorFlow/Keras ecosystem and general data preprocessing best practices."

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            input_desc=description.get('input_data', ''),
            output_desc=description.get('output_data', ''),
            data_file_desc=json.dumps(description.get('data file description', {})),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            preprocessing_guideline=json.dumps(preprocessing_guideline, indent=2),
            target_info=json.dumps(target_info, indent=2),
            tutorials_section=tutorials_section
        )

        if previous_code and error_message:
            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The previously generated code failed with an error.

### Previous Code:
```python
{previous_code}
```

### Error Message:
```
{error_message}
```

## FIX INSTRUCTIONS:
1. Analyze the error message and the previous code carefully.
2. Generate a new, complete, and corrected version of the Python code that resolves the issue.
3. Ensure the corrected code adheres to all the original requirements.
4. If the error indicates missing modules (ModuleNotFoundError/ImportError), wrap imports in try/except and, in the except block, use subprocess to install the missing package (e.g., `[sys.executable, '-m', 'pip', 'install', '<package>']` with `check=True`), then retry the import.

Generate the corrected Python code:
"""
            prompt += retry_context
        
        self.manager.save_and_log_states(prompt, "preprocessing_coder_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "preprocessing_code_response.py")
        return code
