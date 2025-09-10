# src/iML/prompts/description_analyzer_prompt.py
import json
import os
from .base_prompt import BasePrompt
from typing import Dict, Any

class DescriptionAnalyzerPrompt(BasePrompt):

    def default_template(self) -> str:
        return """
You are an expert AI assistant specializing in analyzing Kaggle competition descriptions. Your task is to read the provided text and extract key information into a specific JSON structure.
The output MUST be a valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.
CRITICAL CONSTRAINTS:
- Use the DIRECTORY STRUCTURE as the single source of truth for concrete files/folders/links. Do NOT invent any filenames or paths that are not explicitly visible in the tree below or in its CSV summary section.
- The natural-language DESCRIPTION is context-only to infer goals and task type; do not rely on it to guess missing files.
- For "data file description" keys and for "link to the dataset" entries, include ONLY items that appear in the provided directory structure or CSV summary. If uncertain, omit.
Extract the following information:
- "name": Dataset name
- "input_data": A description of the primary input data for the model.
- "output_data": A description of the expected output format from the model.
- "task": A summary of the main objective or task of the competition.
- "task_type": One of ["text_classification","image_classification","tabular_classification","tabular_regression","seq2seq","ner","qa","unknown"] inferred from the description and directory structure.
- "data file description": A dictionary where the keys are relative path to the file (e.g., "train.csv", "test/test.csv") and the values are their descriptions.
- "link to the dataset": A list containing the filenames and folders of the core data files (like train, test, sample submission). Do NOT invent or guess full paths. Return the relative path from the input_data_folder only, do not contain the input_data_folder dir.
## EXAMPLE:
### INPUT TEXT:

\"\"\"
Welcome to the 'Paddy Disease Classification' challenge! The goal is to classify diseases in rice paddy images. The input data consists of images of rice plants (JPG files) from the `train_images` folder. Your model should output a class label for one of ten possible diseases. The dataset includes `train.csv` which maps image IDs to their labels, `test_images` for prediction, and `sample_submission.csv` for the required format.
\"\"\"

### OUTPUT JSON:

{{
    "name": "paddy_disease_classification",
    "input_data": "The input data consists of images of rice plants (JPG files).",
    "output_data": "The model should output a class label corresponding to one of ten possible diseases.",
    "task": "The main goal is to build a model that can classify diseases in rice paddy images.",
    "task_type": "image_classification",
    "data file description": {{
        "train.csv": "Maps image IDs to their respective disease labels.",
        "train_images": "A folder containing the training images as JPG files.",
        "test_images": "A folder containing the test images for which predictions are required.",
        "sample_submission.csv": "An example file showing the required submission format."
    }},
    "link to the dataset": ["train.csv", "train_images", "test_images", "sample_submission.csv"]
}}
## END OF EXAMPLE. NOW, PROCESS THE FOLLOWING TEXT:
### INPUT TEXT:
\"\"\"
{description}
{directory_structure}
\"\"\"
### OUTPUT JSON:
"""

    # Fixed build method to be correct
    def build(self, description: str, directory_structure: str) -> str:
        """
        Build complete prompt from template and input values.
        """
        # Keep a copy of the last directory structure text to enable strict filtering in parse()
        self._last_directory_structure_text = directory_structure
        prompt = self.template.format(
            description=description,
            directory_structure=directory_structure
        )
        self.manager.save_and_log_states(
            content=prompt, 
            save_name="description_analyzer_prompt.txt"
        )
        return prompt

    # Fixed parse method to be correct
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON object.
        """
        try:
            # Clean and parse JSON string from response
            clean_response = response.strip().replace("```json", "").replace("```", "")
            parsed_response = json.loads(clean_response)
        except json.JSONDecodeError as e:
            # dùng logging toàn cục nếu bạn đã bỏ self.manager.logger
            try:
                import logging
                logging.error(f"Failed to parse JSON from LLM response: {e}")
            except Exception:
                pass
            parsed_response = {"error": "Invalid JSON response from LLM", "raw_response": response}

        # Derive allowed relative paths strictly from the provided directory structure tree and CSV summary
        allowed_rel_paths = set()
        try:
            tree_text = getattr(self, "_last_directory_structure_text", "") or ""
            lines = tree_text.splitlines()
            # Parse tree section (above the summary delimiter of ===)
            path_stack = []  # list of names representing current path
            in_summary = False
            for line in lines:
                if line.strip().startswith("==="):
                    in_summary = True
                if not in_summary:
                    # Look for connectors
                    if "├── " in line or "└── " in line:
                        # split at the last connector occurrence to be robust
                        connector_idx = max(line.rfind("├── "), line.rfind("└── "))
                        prefix = line[:connector_idx]
                        name = line[connector_idx + 4:].strip()
                        if name == "...":
                            continue
                        # depth: each indent unit is 4 chars ("│   " or "    ")
                        depth = len(prefix) // 4
                        # adjust stack to depth
                        if depth <= 0:
                            path_stack = []
                        else:
                            path_stack = path_stack[:depth]
                        # build current relative path
                        rel = "/".join([*path_stack, name]) if path_stack else name
                        # normalize
                        rel = rel.replace("\\", "/").strip("/")
                        if rel:
                            allowed_rel_paths.add(rel)
                        # push to stack as current node
                        if depth == 0:
                            path_stack = [name]
                        else:
                            # ensure length == depth then append
                            if len(path_stack) < depth:
                                path_stack = path_stack + [name]
                            elif len(path_stack) == depth:
                                path_stack.append(name)
                            else:
                                # already trimmed above; append
                                path_stack.append(name)
                else:
                    # CSV summary lines contain: "Structure of file: {rel_path}"
                    marker = "Structure of file: "
                    if marker in line:
                        rel = line.split(marker, 1)[1].strip()
                        rel = rel.replace("\\", "/").strip("/")
                        if rel:
                            allowed_rel_paths.add(rel)
        except Exception:
            # Fail open: if parsing the tree fails, we won't filter by tree-only list here
            allowed_rel_paths = allowed_rel_paths

        # Enforce: keep only entries that exist in allowed_rel_paths; also verify existence on disk to be safe
        dataset_path = self.manager.input_data_folder

        # Filter data file description keys
        if isinstance(parsed_response.get("data file description"), dict):
            filtered_dfd = {}
            for k, v in parsed_response["data file description"].items():
                rel = (k or "").replace("\\", "/").strip("/")
                abs_path = os.path.join(dataset_path, rel)
                if (rel in allowed_rel_paths) and os.path.exists(abs_path):
                    filtered_dfd[rel] = v
            parsed_response["data file description"] = filtered_dfd

        # Filter link to the dataset and then convert to full paths for downstream
        if isinstance(parsed_response.get("link to the dataset"), list):
            filtered_links = []
            seen = set()
            for item in parsed_response["link to the dataset"]:
                rel = (str(item) if item is not None else "").replace("\\", "/").strip("/")
                abs_path = os.path.join(dataset_path, rel)
                if (rel in allowed_rel_paths) and os.path.exists(abs_path) and rel not in seen:
                    filtered_links.append(rel)
                    seen.add(rel)
            # Convert to full paths (existing downstream behavior)
            full_paths = [os.path.join(dataset_path, rel).replace("\\", "/") for rel in filtered_links]
            parsed_response["link to the dataset"] = full_paths

        self.manager.save_and_log_states(
            content=json.dumps(parsed_response, indent=2), 
            save_name="description_analyzer_response.json"
        )
        return parsed_response
