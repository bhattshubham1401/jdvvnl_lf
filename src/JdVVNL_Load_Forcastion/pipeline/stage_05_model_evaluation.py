from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.components.model_evaluation import ModelEvaluation
from src.JdVVNL_Load_Forcastion.config.configuration import ConfigurationManager

STAGE_NAME = "Model evaluation stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.info(f"Error occur in {STAGE_NAME} ")
        raise e
