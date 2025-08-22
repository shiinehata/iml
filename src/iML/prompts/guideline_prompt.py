# src/iML/prompts/guideline_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

def _create_variables_summary(variables: dict) -> dict:
    """Create a concise summary for variables in the profile."""
    summary = {}
    for var_name, var_details in variables.items():
        summary[var_name] = {
            "type": var_details.get("type"),
            "n_unique": var_details.get("n_unique"),
            "p_missing": var_details.get("p_missing"),
            "mean": var_details.get("mean"),
            "std": var_details.get("std"),
            "min": var_details.get("min"),
            "max": var_details.get("max"),
        }
    return summary

class GuidelinePrompt(BasePrompt):
    """
    Prompt handler to create guidelines for AutoML pipeline.
    """

    def default_template(self) -> str:
        """Default template to request LLM to create guidelines."""
        return """You are an expert Machine Learning architect. Your task is to analyze the provided dataset information and create a specific, actionable, and justified guideline for an AutoML pipeline.
## Dataset Information:
- Dataset: {dataset_name}
- Task: {task_desc}
- Size: {n_rows:,} rows, {n_cols} columns
- Key Quality Alerts: {alerts}
- Output format: {output_data}

## Variables Analysis Summary:
```json
{variables_summary_str}
```

## Guideline Generation Principles & Examples
Your response must be guided by the following principles. Refer to these examples to understand the required level of detail.

BE SPECIFIC AND ACTIONABLE: Your recommendations must be concrete actions.
- Bad (Generic): "Handle missing values"
- Good (Specific): "Impute 'Age' with the median"

JUSTIFY YOUR CHOICES INTERNALLY: Even if the final JSON does not include every reasoning detail, your internal decision process must be sound, based on the data properties.

IT IS ACCEPTABLE TO OMIT: If a step is not necessary, provide an empty list or null for that key in the JSON output.

High-Quality Examples:

Example 1: Feature Engineering for a DateTime column
For a DateTime column like 'transaction_date', a good feature_engineering list would be ["Extract 'month' from 'transaction_date'", "Extract 'day_of_week' from 'transaction_date'"].

Example 2: Handling High Cardinality Categorical Data
For a categorical column 'product_id' with over 100 unique values, a good recommendation is ["Apply frequency encoding to 'product_id'"].

Example 3: Handling Missing Numerical Data
For a numeric column 'income' with 25% missing values and a skewed distribution, a good recommendation is ["Impute 'income' with its median"].

Before generating the final JSON, consider:
1. Identify the target variable and task type (classification, regression, etc.).
2. Review each variable's type, statistics, and potential issues.
3. Decide on specific actions for data preprocessing and modeling based on the data's properties.
4. If using pretrained models, choose the most appropriate ones.
5. Compile these specific actions into the required JSON format.

Output Format: Your response must be in the JSON format below:
Provide your response in JSON format. An empty list or null is acceptable for recommendations if not applicable.

IMPORTANT: Ensure the generated JSON is perfectly valid.
- All strings must be enclosed in double quotes.
- All backslashes inside strings must be properly escaped.
- There should be no unescaped newline characters within a string value.
- Do not include comments within the JSON output.

{{
    "target_identification": {{
        "target_variable": "identified_target_column_name",
        "reasoning": "explanation for target selection",
        "task_type": "classification/regression/etc"
    }},
    "preprocessing": {{
        "data_cleaning": ["specific step 1", "specific step 2"],
        "feature_engineering": ["specific technique 1", "specific technique 2"],
        "missing_values": ["strategy 1", "strategy 2"],
        "feature_selection": ["method 1", "method 2"],
        "data_splitting": {{"train": 0.8, "val": 0.2, "strategy": "appropriate strategy"}}
    }},
    "modeling": {{
        "recommended_algorithms": ["algorithm 1", "algorithm 2"],
        "model_selection": ["model_name1", "model_name2"],
        "cross_validation": {{"method": appropriate method, "scoring": appropriate metric}}
    }},
    "evaluation": {{
        "metrics": ["metric 1", "metric 2"],
        "validation_strategy": ["approach 1", "approach 2"],
        "performance_benchmarking": ["baseline 1", "baseline 2"],
        "result_interpretation": ["interpretation 1", "interpretation 2"]
    }}
}}"""

    def build(self, description_analysis: Dict[str, Any], profiling_result: Dict[str, Any]) -> str:
        """Build prompt from analysis and profiling results."""
        task_info = description_analysis
        
        # Find key of train file in profiling results
        train_key = None
        for key in profiling_result.get('summaries', {}).keys():
            if 'test' not in key.lower() and 'submission' not in key.lower():
                train_key = key
                break
        
        if not train_key:
             # If not found, take the first key as default
            train_key = next(iter(profiling_result.get('summaries', {})), None)

        train_summary = profiling_result.get('summaries', {}).get(train_key, {})
        train_profile = profiling_result.get('profiles', {}).get(train_key, {})

        n_rows = train_summary.get('n_rows', 0)
        n_cols = train_summary.get('n_cols', 0)
        alerts = train_profile.get('alerts', [])
        variables = train_profile.get('variables', {})
        variables_summary_str = json.dumps(_create_variables_summary(variables), indent=2, ensure_ascii=False)
        dataset_name = task_info.get('name', 'N/A')
        task_desc = task_info.get('task', 'N/A')
        output_data = task_info.get('output_data', 'N/A')

        prompt = self.template.format(
            dataset_name=dataset_name,
            task_desc=task_desc,
            n_rows=n_rows,
            n_cols=n_cols,
            alerts=alerts[:3] if alerts else 'None',
            variables_summary_str=variables_summary_str,
            output_data=output_data
        )
        
        self.manager.save_and_log_states(prompt, "guideline_prompt.txt")
        return prompt

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            parsed_response = json.loads(response.strip().replace("```json", "").replace("```", ""))
        except json.JSONDecodeError as e:
            self.manager.logger.error(f"Failed to parse JSON from LLM response for guideline: {e}")
            parsed_response = {"error": "Invalid JSON response from LLM", "raw_response": response}
        
        self.manager.save_and_log_states(
            json.dumps(parsed_response, indent=4, ensure_ascii=False), 
            "guideline_response.json"
        )
        return parsed_response
