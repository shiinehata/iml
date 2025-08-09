import uuid
from datetime import datetime
from pathlib import Path
import logging

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

from .core.manager import Manager
from .utils.rich_logging import configure_logging

def run_automl_pipeline(input_data_folder: str, output_folder: str = None, config_path: str = "configs/default.yaml"):
    """
    Main function to set up the environment and run the entire pipeline.
    """
    # 1. Create the output directory if one is not provided
    if output_folder is None:
        project_root = Path(__file__).parent.parent.parent # Points to the project root
        working_dir = project_root / "runs"
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_uuid = uuid.uuid4().hex[:8]
        folder_name = f"run_{current_datetime}_{random_uuid}"
        output_path = working_dir / folder_name
    else:
        output_path = output_folder

    # Ensure output_path is a Path object and the directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load the configuration file
    config = OmegaConf.load(config_path)

    # 3. Configure the logging system FIRST
    # This is critical to ensure all subsequent logs respect the verbosity level
    configure_logging(output_dir=output_dir, verbosity=config.verbosity)
    
    logger.info(f"Project running. Output will be saved to: {output_dir.resolve()}")
    logger.info(f"Loaded configuration from: {config_path}")
    logger.debug(f"Full configuration details: {OmegaConf.to_yaml(config)}")

    # 4. Initialize the Manager with the prepared settings
    manager = Manager(
        input_data_folder=input_data_folder,
        output_folder=str(output_dir),  # Pass as string for consistency
        config=config,
    )

    # 5. Start the pipeline run
    manager.run_pipeline()

    manager.report_token_usage()
    logger.brief(f"output saved in {output_dir}.")
    manager.cleanup()