from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.entity.config_entity import DataIngestionConfig
from src.JdVVNL_Load_Forcastion.utils.common import get_data_from_api_query


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiateDataIngestion(self):
        try:
            data = get_data_from_api_query()
            logger.info("Reading Completed from MongoDB")
            logger.info("Writing data in Parquet file")
            data.to_parquet(self.config.local_data_file1, index=False, engine='auto')

        except Exception as e:
            print(f"Error in Ingestion Process: {e}")