import logging
import os
import uuid
import subprocess
from pathlib import Path
from typing import List

from ..agents import (
    DescriptionAnalyzerAgent,
    ProfilingAgent,
    GuidelineAgent,
    PreprocessingCoderAgent,
    ModelingCoderAgent,
    AssemblerAgent,
    FeedbackAgent,
    CandidateGeneratorAgent,
    CandidateSelectorAgent,
)
from ..llm import ChatLLMFactory

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)


class Manager:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
    ):
        """Initialize Manager with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        self.time_step = -1

        # Store required paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config

        self.description_analyzer_agent = DescriptionAnalyzerAgent(
            config=config,
            manager=self,
            llm_config=self.config.description_analyzer,
        )
        self.profiling_agent = ProfilingAgent(
            config=config,
            manager=self,
        )
        self.guideline_agent = GuidelineAgent(
            config=config,
            manager=self,
            llm_config=self.config.guideline_generator,
        )
        self.preprocessing_coder_agent = PreprocessingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.preprocessing_coder,
        )
        self.modeling_coder_agent = ModelingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.modeling_coder,
        )
        self.assembler_agent = AssemblerAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,
        )
        self.feedback_agent = FeedbackAgent(
            config=config,
            manager=self,
            llm_config=self.config.feedback,
        )
        self.candidate_generator_agent = CandidateGeneratorAgent(
            config=config,
            manager=self,
            llm_config=self.config.candidate_generator,
        )
        self.candidate_selector_agent = CandidateSelectorAgent(
            config=config,
            manager=self,
            llm_config=self.config.candidate_selector,
        )

        # Initialize prompts
        self.generate_initial_prompts()

        self.user_inputs: List[str] = []
        self.error_messages: List[str] = []
        self.error_prompts: List[str] = []
        self.python_codes: List[str] = []
        self.python_file_paths: List[str] = []
        self.bash_scripts: List[str] = []
        self.tutorial_retrievals: List[str] = []
        self.tutorial_prompts: List[str] = []

    def run_pipeline(self):
        """Run the entire pipeline from description analysis to code generation."""
        self.time_step = 0

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return
        logger.info(f"Analysis result: {analysis_result}")

        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return
        
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3: Run guideline agent
        guideline = self.guideline_agent()
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return
        
        self.guideline = guideline
        logger.info("Guideline generated successfully.")

        # Step 4: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent()
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return

        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 5: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent()
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return
            
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully (not yet validated).")

        # Step 6: Run Assembler Agent to assemble, finalize, and run the code
        assembler_result = self.assembler_agent()
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return
        
        self.assembled_code = assembler_result.get("code")
        logger.info(f"Initial script generated and executed successfully.")
        
        """
        # --- SELF-IMPROVEMENT LOOP ---
        logger.info("Starting self-improvement loop...")

        # Step 7: Run Feedback Agent
        feedback_result = self.feedback_agent()
        if feedback_result.get("status") == "failed":
            logger.error(f"Feedback generation failed: {feedback_result.get('error')}")
            # This is not a fatal error, we can still proceed with the original code
            logger.warning("Proceeding without code improvement.")
        else:
            self.feedback = feedback_result.get("feedback")
            logger.info("Feedback for improvement generated successfully.")

            # Step 8: Run Candidate Generator Agent
            candidate_result = self.candidate_generator_agent()
            if candidate_result.get("status") == "failed" or not candidate_result.get("candidates"):
                logger.error(f"Candidate generation failed: {candidate_result.get('error')}")
                logger.warning("Proceeding without code improvement.")
            else:
                self.candidates = candidate_result.get("candidates")
                logger.info(f"Generated {len(self.candidates)} valid candidates.")

                # Step 9: Run Candidate Selector Agent
                selector_result = self.candidate_selector_agent()
                if selector_result.get("status") == "failed":
                    logger.error(f"Candidate selection failed: {selector_result.get('error')}")
                    logger.warning("Proceeding without code improvement.")
                else:
                    self.selected_code = selector_result.get("selected_code")
                    logger.info("Best candidate selected. Preparing for final execution.")

                    # Step 10: Final Execution of the selected code
                    logger.info("Executing the improved final script...")
                    final_execution_result = self.execute_code(self.selected_code)
                    if final_execution_result["success"]:
                         logger.info(f"Improved script executed successfully.")
                    else:
                         logger.error(f"Execution of improved script failed. Error: {final_execution_result['stderr']}")
                         logger.warning("The improved script failed. The result from the initial script is still available.")
        """

        logger.info("AutoML pipeline completed successfully!")

    def generate_initial_prompts(self):

        # TODO: remove the hard code for "create_venv" (add in tool registry if need installation)
        asds =1

    @property
    def user_input(self) -> str:
        assert self.time_step >= 0, "No user input because the prompt generator is not stepped yet."
        assert len(self.user_inputs) == self.time_step + 1, "user input is not updated yet"
        return self.user_inputs[self.time_step]

    @property
    def python_code(self) -> str:
        assert self.time_step >= 0, "No python code because the prompt generator is not stepped yet."
        assert len(self.python_codes) == self.time_step + 1, "python code is not updated yet"
        return self.python_codes[self.time_step]

    @property
    def python_file_path(self) -> str:
        assert self.time_step >= 0, "No python file path because the prompt generator is not stepped yet."
        assert len(self.python_file_paths) == self.time_step + 1, "python file path is not updated yet"
        return self.python_file_paths[self.time_step]

    @property
    def previous_python_code(self) -> str:
        if self.time_step >= 1:
            return self.python_codes[self.time_step - 1]
        else:
            return ""

    @property
    def bash_script(self) -> str:
        assert self.time_step >= 0, "No bash script because the prompt generator is not stepped yet."
        assert len(self.bash_scripts) == self.time_step + 1, "bash script is not updated yet"
        return self.bash_scripts[self.time_step]

    @property
    def previous_bash_script(self) -> str:
        if self.time_step >= 1:
            return self.bash_scripts[self.time_step - 1]
        else:
            return ""

    @property
    def error_message(self) -> str:
        assert self.time_step >= 0, "No error message because the prompt generator is not stepped yet."
        assert len(self.error_messages) == self.time_step + 1, "error message is not updated yet"
        return self.error_messages[self.time_step]

    @property
    def previous_error_message(self) -> str:
        if self.time_step >= 1:
            return self.error_messages[self.time_step - 1]
        else:
            return ""

    @property
    def error_prompt(self) -> str:
        assert self.time_step >= 0, "No error prompt because the prompt generator is not stepped yet."
        assert len(self.error_prompts) == self.time_step + 1, "error prompt is not updated yet"
        return self.error_prompts[self.time_step]

    @property
    def previous_error_prompt(self) -> str:
        if self.time_step >= 1:
            return self.error_prompts[self.time_step - 1]
        else:
            return ""

    @property
    def all_previous_error_prompts(self) -> str:
        if self.time_step >= 1:
            return "\n\n".join(self.error_prompts[: self.time_step])
        else:
            return ""

    @property
    def tutorial_prompt(self) -> str:
        assert self.time_step >= 0, "No tutorial prompt because the prompt generator is not stepped yet."
        assert len(self.tutorial_prompts) == self.time_step + 1, "tutorial prompt is not updated yet"
        return self.tutorial_prompts[self.time_step]

    @property
    def previous_tutorial_prompt(self) -> str:
        if self.time_step >= 1:
            return self.tutorial_prompts[self.time_step - 1]
        else:
            return ""

    @property
    def tutorial_retrieval(self) -> str:
        assert self.time_step >= 0, "No tutorial retrieval because the prompt generator is not stepped yet."
        assert len(self.tutorial_retrievals) == self.time_step + 1, "tutorial retrieval is not updated yet"
        return self.tutorial_retrievals[self.time_step]

    @property
    def previous_tutorial_retrieval(self) -> str:
        if self.time_step >= 1:
            return self.tutorial_retrievals[self.time_step - 1]
        else:
            return ""

    @property
    def iteration_folder(self) -> str:
        if self.time_step >= 0:
            iter_folder = os.path.join(self.output_folder, f"generation_iter_{self.time_step}")
        else:
            iter_folder = os.path.join(self.output_folder, "initialization")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    def set_initial_user_input(self, need_user_input, initial_user_input):
        self.need_user_input = need_user_input
        self.initial_user_input = initial_user_input

    def step(self):
        """Step the prompt generator forward."""
        self.time_step += 1

        user_input = self.initial_user_input
        # Get per iter user inputs if needed
        if self.need_user_input:
            if self.time_step > 0:
                logger.brief(
                    f"[bold green]Previous iteration info is stored in:[/bold green] {os.path.join(self.output_folder, f'iteration_{self.time_step - 1}')}"
                )
            else:
                logger.brief(
                    f"[bold green]Initialization info is stored in:[/bold green] {os.path.join(self.output_folder, 'initialization')}"
                )
            if user_input is None:
                user_input = ""

        assert len(self.user_inputs) == self.time_step
        self.user_inputs.append(user_input)

        if self.time_step > 0:
            previous_error_prompt = self.error_analyzer()

            assert len(self.error_prompts) == self.time_step - 1
            self.error_prompts.append(previous_error_prompt)

        retrieved_tutorials = self.retriever()
        assert len(self.tutorial_retrievals) == self.time_step
        self.tutorial_retrievals.append(retrieved_tutorials)

        tutorial_prompt = self.reranker()
        assert len(self.tutorial_prompts) == self.time_step
        self.tutorial_prompts.append(tutorial_prompt)

    def write_code_script(self, script, output_code_file):
        with open(output_code_file, "w") as file:
            file.write(script)

    def execute_code(self, code_to_execute: str, phase_name: str, attempt: int) -> dict:
        """
        Executes a string of Python code in a subprocess.

        Args:
            code_to_execute: The Python code to run.
            phase_name: The name of the phase (e.g., "preprocessing", "assembler").
            attempt: The retry attempt number.

        Returns:
            A dictionary with execution status, stdout, and stderr.
        """
        # Create a temporary file to write the code to
        # Use a unique name for each execution to avoid conflicts
        temp_script_name = f"{phase_name}_attempt_{attempt}_{uuid.uuid4().hex[:8]}.py"
        temp_script_path = os.path.join(self.iteration_folder, temp_script_name)
        
        self.write_code_script(code_to_execute, temp_script_path)

        logger.info(f"Executing code from temporary file: {temp_script_path}")

        try:
            # Execute the script using subprocess
            # The script should be executed from a directory where it can access the data
            # Assuming the data paths in the script are relative to the input folder's parent
            working_dir = str(Path(self.input_data_folder).parent)
            
            process = subprocess.run(
                ["python", temp_script_path],
                capture_output=True,
                text=True,
                check=False,  # Do not raise exception on non-zero exit code
                cwd=working_dir
            )

            stdout = process.stdout
            stderr = process.stderr

            self.save_and_log_states(stdout, f"exec_stdout_{os.path.basename(temp_script_path)}.txt")
            self.save_and_log_states(stderr, f"exec_stderr_{os.path.basename(temp_script_path)}.txt")

            if process.returncode == 0:
                logger.info("Code executed successfully.")
                return {"success": True, "stdout": stdout, "stderr": stderr}
            else:
                logger.error(f"Code execution failed with return code {process.returncode}.")
                # Combine stdout and stderr for a complete error context
                full_error = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                return {"success": False, "stdout": stdout, "stderr": full_error}

        except Exception as e:
            logger.error(f"An exception occurred during code execution: {e}")
            return {"success": False, "stdout": "", "stderr": str(e)}


    def update_python_code(self):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        python_code = self.python_coder()

        python_file_path = os.path.join(self.iteration_folder, "generated_code.py")

        self.write_code_script(python_code, python_file_path)

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step

        bash_script = self.bash_coder()

        bash_file_path = os.path.join(self.iteration_folder, "execution_script.sh")

        self.write_code_script(bash_script, bash_file_path)

        self.bash_scripts.append(bash_script)

    def execute_code_old(self):
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", add_uuid=False)
        self.save_and_log_states(stdout, "stdout", add_uuid=False)

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            return False
        elif planner_decision == "FINISH":
            logger.brief(
                f"[bold green]Code generation successful after[/bold green] {self.time_step + 1} [bold green]iterations[/bold green]"
            )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def save_and_log_states(self, content, save_name, add_uuid=False):
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        states_dir = os.path.join(self.output_folder, "states")
        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        logger.brief(message)

    def log_agent_end(self, message: str):
        logger.brief(message)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
