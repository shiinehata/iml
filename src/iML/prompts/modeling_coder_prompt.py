# src/iML/prompts/modeling_coder_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

class ModelingCoderPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for modeling.
    """

    def default_template(self) -> str:
        """Default template to request LLM to generate modeling code."""
        return """
You are an expert ML engineer. Your task is to generate Python code for modeling, which will be combined with the provided preprocessing code.

## CONTEXT
- **Dataset Name**: {dataset_name}
- **Task Description**: {task_desc}
- **File Paths**: {file_paths} (LOAD DATA FROM THESE PATHS)
- **Data File Description**: {data_file_description}
- **Output data format**: {output_data_format} 

## MODELING GUIDELINES:
{modeling_guideline}

## RELEVANT TUTORIALS:
{tutorials_section}

## PREPROCESSING CODE (Do NOT include this in your response):
The following preprocessing code, including a function `preprocess_data(file_paths: dict)`, will be available in the execution environment. You must call it to get the data.
```python
{preprocessing_code}
```

## REQUIREMENTS:
1.  **CRITICAL: USE THE COMPLETE CODE EXAMPLES from the provided tutorials whenever possible.** Copy and adapt the exact model architectures, training loops, and methodologies from the tutorial code blocks.
2.  **Generate COMPLETE Python code for the modeling part ONLY.** Do NOT repeat the preprocessing code.
3.  Your code should start with imports EXACTLY as shown in the tutorial code examples whenever possible (e.g., `import tensorflow as tf`, `from tensorflow import keras`).
4.  **CRITICAL: Adapt your function signature to match the data format returned by preprocess_data()** - this could be generators, datasets, or arrays depending on the tutorial approach.
5.  Keep the data loading code of the preprocessing code.
6.  The main execution block (`if __name__ == "__main__":`) must:
    a. Call the existing `preprocess_data()` function to get the datasets in the format specified by preprocessing.
    b. Call your training and prediction function with the appropriate data format.
    c. Save the predictions to a `submission.csv` file. The format should typically be two columns: an identifier column and the prediction column.
7.  **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error to stderr and **exit with a non-zero status code** (`sys.exit(1)`).
8.  **STRONGLY ENCOURAGED: Follow tutorial code structure for algorithm choice when available. Tutorials provide proven approaches for your task type.**
9.  **ENCOURAGED: Use the SAME model architectures, training approaches, and code patterns demonstrated in the tutorials when they match your data format and task requirements.**
10. Do not use extensive hyperparameter tuning unless specified. Keep the code efficient.
11. Limit comments in the code.
12. The submission file must have the same structure (number of columns) as the sample submission file provided in the dataset, but may have different ID. You have to use the test data to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
13. Your final COMPLETE Python code should have only ONE main function. If there are duplicate main function, remove the duplicates and keep only one main function.
14. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
15. **ENCOURAGED: When tutorials provide COMPLETE CODE EXAMPLES that match your data format, prioritize using those exact model structures, training patterns, and approaches. Adapt as needed for your specific data format.**


## IMPORTANT: 
- **EXAMINE the preprocessing code carefully** to understand the data format it returns (arrays, generators, tf.data.Dataset, etc.)
- **ADAPT your modeling code** to work with that specific data format
- **FOLLOW tutorial patterns** when they are compatible with your data format
- **If tutorials use a different data format**, adapt the tutorial approach to work with your preprocessing output

## CODE STRUCTURE GUIDELINES:
```python
# Adapt this structure based on your actual preprocessing output format
# Example for array-based data (adapt as needed):

import sys
import os
# Import libraries that match your tutorial examples and data format

{preprocessing_code}

def train_and_predict(data_input):
    # Adapt function signature and logic based on preprocessing output format
    # Use tutorial model architectures when compatible
    # Return predictions in appropriate format
    pass

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # 1. Get data from preprocessing (format depends on preprocessing implementation)
        data_output = preprocess_data(file_paths)
        
        # 2. Adapt to the actual format returned by preprocess_data
        # This could be: arrays, generators, tf.data.Dataset, etc.
        
        # 3. Train and predict using compatible approach
        predictions = train_and_predict(data_output)
        
        # 4. Create submission file with appropriate format
        # submission_df = pd.DataFrame(...) 
        # submission_df.to_csv("submission.csv", index=False)

        print("Modeling script executed successfully and submission.csv created!")

    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```
"""

    def build(self, guideline: Dict, description: Dict, preprocessing_code: str, previous_code: str = None, error_message: str = None, tutorials_content: str = "") -> str:
        """Build prompt to generate modeling code."""
        
        modeling_guideline = guideline.get('modeling', {})
        
        # Format tutorials section
        if tutorials_content.strip():
            tutorials_section = f"""The following tutorials provide COMPLETE CODE EXAMPLES and proven approaches for your task type:

{tutorials_content}

**TUTORIAL GUIDANCE (STRONGLY ENCOURAGED):**
1. **EXAMINE the tutorial code examples** above for proven model architectures and training approaches compatible with your task
2. **ADAPT tutorial patterns** to work with your specific data format (as returned by preprocess_data)
3. **PRIORITIZE tutorial libraries and methods** when they are compatible with your preprocessing output format
4. **USE tutorial coding structures** as a foundation and modify as needed for your data format
5. **LEVERAGE tutorial insights** for model selection, training strategies, and best practices
6. If tutorial data format differs from your preprocessing output, adapt the tutorial approach to your format"""
        else:
            tutorials_section = "No specific tutorials found for this task type. Use appropriate ML frameworks and general machine learning best practices based on your data format."

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            data_file_description=description.get('data file description', 'N/A'),
            output_data_format=description.get('output_data', 'N/A'),
            modeling_guideline=json.dumps(modeling_guideline, indent=2),
            preprocessing_code=preprocessing_code,
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
2. Fix the specific issue that caused the error.
3. Ensure your code correctly uses the data returned by the `preprocess_data` function.
4. Generate a new, complete, and corrected version of the Python code that resolves the issue.
5. Adhere to all original requirements.

Generate the corrected Python code:
"""
            prompt += retry_context
        
        self.manager.save_and_log_states(prompt, "modeling_coder_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "modeling_code_response.py")
        return code
