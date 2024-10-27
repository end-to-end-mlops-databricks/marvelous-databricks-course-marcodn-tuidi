import mlflow
from databricks.connect import DatabricksSession
from src.personality_types.config import ProjectConfig
from src.personality_types.personality_model_prob import PersonalityModelProb
from src.utils.logger_utils import set_logger

spark = DatabricksSession.builder.getOrCreate()

logger = set_logger()

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

logger.info("Load pretrained model")
run_id = mlflow.search_runs(
    experiment_names=["/Users/marco.dinardo@tuidi.it/personality-types"],
    filter_string="tags.branch='week_2'",
).run_id[0]

model_run = f"runs:/{run_id}/randomforest-pipeline-model"

model = mlflow.sklearn.load_model(model_run)

personality_model_prob = PersonalityModelProb(model, config)

git_sha = "test"

run_tags = {"git_sha": git_sha, "branch": "week_2"}

model_version = personality_model_prob.log_model(
    spark,
    "/Users/marco.dinardo@tuidi.it/personality-types-prob",
    run_tags,
    "best_model",
)
