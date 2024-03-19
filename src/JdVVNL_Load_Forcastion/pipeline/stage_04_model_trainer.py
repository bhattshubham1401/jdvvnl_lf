from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.components.model_trainer import ModelTrainer
from src.JdVVNL_Load_Forcastion.config.configuration import ConfigurationManager

STAGE_NAME = "Model Trainer"


class ModelTrainingPipeline:
    def __int__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            ModelTrainerConfig = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=ModelTrainerConfig)
            model_trainer.train()

        except Exception as e:
            logger.info(f"Error Occur in Model Training Pipeline {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        data = ModelTrainingPipeline()
        data.main()
        logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    except Exception as e:
        logger.info(f"Error occur in Stage {STAGE_NAME}. ")
