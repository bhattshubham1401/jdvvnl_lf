from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.JdVVNL_Load_Forcastion.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.JdVVNL_Load_Forcastion.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.JdVVNL_Load_Forcastion.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from src.JdVVNL_Load_Forcastion.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

logger.info("We are printing the logs here!!!")

# STAGE_NAME = "Data Ingestion"
# try:
#     logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info("-----------Stage {} Completed-----------".format(STAGE_NAME))
#
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# STAGE_NAME = "Data Validation"
#
# try:
#     logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
#     obj = DataValidationTrainingPipeline()
#     obj.main()
#     logger.info("-----------Stage {} Completed------------".format(STAGE_NAME))
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Data Transformation"

try:
    logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info("-----------Stage {} Completed------------".format(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info("-----------Stage {} Completed------------".format(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model evaluation stage"
try:
    logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info("-----------Stage {} Completed------------".format(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e
