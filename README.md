# iML: Intelligent Machine Learning

`iML` constitutes an intelligent framework for Automated Machine Learning (AutoML), engineered for the comprehensive automation of the entire workflow associated with resolving machine learning problems. This encompasses all phases from initial requirement analysis and data preprocessing through to model selection, code generation, and subsequent self-improvement protocols.

The project's architecture is predicated upon a system of intelligent **agents**. Each agent is functionally specialized for a discrete task and is augmented by Large Language Models (LLMs) to facilitate informed decision-making throughout the operational pipeline.

---
## Key Features

* **End-to-End Automation**: The framework facilitates a complete, end-to-end automation of the analytical process, commencing with initial problem analysis and culminating in the generation of final executable code.

* **Intelligent Agent Architecture**: The operational workflow is partitioned into a series of independent agents, with each agent assigned responsibility for a distinct stage of the process:

  * `DescriptionAnalyzerAgent`: Responsible for the analysis of the problem specification.

  * `ProfilingAgent`: Tasked with the analysis and summarization of input data.

  * `GuidelineAgent`: Proposes procedural steps and strategic approaches for the solution.

  * `PreprocessingCoderAgent`: Generates code pertinent to data preprocessing.

  * `ModelingCoderAgent`: Generates code for the training of machine learning models.

  * `AssemblerAgent`: Assembles and finalizes the source code into an executable form.

  * **Self-Improvement Loop**: A self-improvement cycle is incorporated, comprising the `FeedbackAgent`, `CandidateGeneratorAgent`, and `CandidateSelectorAgent`, to enable the automatic debugging and optimization of the generated solution.

* **Multi-LLM Provider Support**: A notable feature is the capacity for seamless integration with a plurality of Large Language Model (LLM) providers, including but not limited to OpenAI, Azure, Anthropic, and Bedrock.

* **Structured Logging and Artifacts**: Each execution instance is archived within a dedicated directory located in the `runs/` folder, a methodology that facilitates systematic tracking, comparison, and reuse of procedural outcomes.

---
## How to Run

The initiation of the iML AutoML pipeline is accomplished via the execution of the `run.py` script, which accepts several command-line arguments.

### System Prerequisites

* An installation of Python, version 3.11.

* All requisite dependencies, as enumerated within a `requirements.txt` file, should such a file be provided.

* Properly configured environment variables corresponding to the selected Large Language Model (LLM) provider (e.g., `OPENAI_API_KEY`).

### Execution Syntax

```
python run.py -i <path_to_data_folder> -c <path_to_config_file> -o <path_to_output_folder>

```

### Command-Line Arguments

* `-i` or `--input` (required): Specifies the path to the directory containing the input data (e.g., `path/to/your/dataset`).

* `-c` or `--config` (optional): Specifies the path to the YAML configuration file. If this argument is omitted, the default configuration located at `configs/default.yaml` will be utilized.

* `-o` or `--output` (optional): Specifies the path to the directory wherein the results are to be saved. In the absence of this argument, a new directory will be programmatically generated within the `runs/` folder, named according to the format `run_<datetime>_<uuid>`.

### Illustrative Example

```
python run.py -i ./datasets/steel_plate_defect -c configs/default.yaml

```

The execution of the aforementioned command will initiate the AutoML pipeline for the dataset situated at `./datasets/steel_plate_defect`. All resulting artifacts will be stored in a newly created directory within the `runs/` folder.

---
## workflow Project Workflow

The operational methodology of `iML` is characterized by a structured, multi-stage process, which is orchestrated by the `Manager` module located at `src/iML/core/manager.py`. A detailed breakdown of this workflow is provided herein.

**High-Level Workflow Diagram:**

1. **Initialization (`run.py` & `main_runner.py`)**: The process is initiated through the user's execution of the `run.py` script with the requisite input arguments. The `main_runner.py` module subsequently receives these arguments, establishes a unique output directory for the current execution, and configures the system-wide logging mechanism. A `Manager` instance is then created to orchestrate the entirety of the pipeline.

2. **Phase 1: Analysis and Planning**:

   * **`DescriptionAnalyzerAgent`**: This agent performs a semantic analysis of the problem description, typically sourced from a `description.txt` file, to ascertain the primary objective, classify the task type (e.g., classification, regression), and identify any other pertinent requirements.

   * **`ProfilingAgent`**: This agent conducts an automated scan of the provided data files (e.g., `.csv`) and generates a summary report detailing the data structure, column schemas, data types, and a representative sample of rows.

   * **`GuidelineAgent`**: Synthesizing the outputs from the problem and data analyses, this agent formulates a detailed strategic plan. This plan outlines necessary preprocessing steps, suggests potential algorithms for consideration, and recommends appropriate feature engineering techniques.

3. **Phase 2: Initial Code Generation**:

   * **`PreprocessingCoderAgent`**: In accordance with the established plan, this agent generates a Python script dedicated exclusively to data preprocessing. This script undergoes independent validation to ensure freedom from syntactical and fundamental logical errors.

   * **`ModelingCoderAgent`**: Subsequently, this agent generates the code required for model training, evaluation, and prediction.

   * **`AssemblerAgent`**: This agent is responsible for assembling the preprocessing and modeling code segments into a single, complete, and executable script (`final_executable_code.py`). It also injects necessary boilerplate code for tasks such as data loading and results persistence, and thereafter executes the script. Should errors arise, it is designed to attempt remedial actions over several retries.

4. **Phase 3: Self-Improvement Loop**:

   * **`FeedbackAgent`**: Following a successful initial execution, this agent performs an analysis of the generated code and its output, comparing them against the original strategic plan to provide constructive feedback. Such feedback might include observations like, "The code lacks a data standardization step," or suggestions such as, "Consider the integration of a LightGBM model to potentially enhance predictive accuracy."

   * **`CandidateGeneratorAgent`**: Acting upon the provided feedback, this agent generates multiple alternative code versions, or `candidates`, each representing an attempt to improve upon the original script. The executability of every generated candidate is rigorously tested.

   * **`CandidateSelectorAgent`**: The valid candidates are then subjected to an evaluation process. This agent selects the optimal version based on performance metrics or other predefined criteria.

   * The selected code is then designated as the final executable for the run.

5. **Completion**: All artifacts produced throughout the process—including code generated at each discrete step (`preprocessing_code_response.py`, `modeling_code_response.py`), temporary execution files (`temp_exec_*.py`), and the final outputs (`submission.csv`, `final_executable_code.py`)—are systematically archived within the run's dedicated output directory.

The resultant directory structure, formatted as `runs/run_<timestamp>_<uuid>/`, serves as a direct artifact of this procedural workflow, wherein each subdirectory and file meticulously logs the state of the system at every discrete step.