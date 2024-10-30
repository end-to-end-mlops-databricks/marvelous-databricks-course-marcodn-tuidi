from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """
    Configuration class for managing project-specific settings, such as feature
    lists, target variables, database details, and model hyperparameters.
    """

    num_features: List[str]
    cat_features: List[str]
    raw_target: str
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters

    @classmethod
    def from_yaml(cls, config_path: str):
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            ProjectConfig: An instance of `ProjectConfig` with data populated
            from the YAML file.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Class method to load configuration data from a dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration data.

        Returns:
            ProjectConfig: An instance of `ProjectConfig` with data populated
            from the provided dictionary.
        """
        return cls(**config_dict)
