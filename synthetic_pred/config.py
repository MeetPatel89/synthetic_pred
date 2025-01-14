from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def configure_paths() -> dict[str, str]:
    """Configure paths for project root and data directories."""
    current_path = Path(__file__).resolve()
    proj_root = str(current_path.parents[1])
    data_dir = f"{proj_root}/data/"
    bronze_dir = f"{data_dir}bronze/"
    return {"proj_root": proj_root, "data_dir": data_dir, "bronze_dir": bronze_dir}


config_paths = configure_paths()
