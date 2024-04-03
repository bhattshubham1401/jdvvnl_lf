import os
from pathlib import Path

from urllib.parse import urlparse

from matplotlib import pyplot as plt

from src.JdVVNL_Load_Forcastion import logger
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback
from datetime import datetime, timedelta
from src.JdVVNL_Load_Forcastion.entity.config_entity import ModelEvaluationConfig
from src.JdVVNL_Load_Forcastion.utils.common import save_json
from src.JdVVNL_Load_Forcastion.components.data_transformation import create_features, add_lags
from src.JdVVNL_Load_Forcastion.utils.common import store_predictions_in_mongodb


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def save_model_as_dict(self, models_dict):
        # Save the model as a dictionary
        joblib.dump(models_dict, self.config.model_path)

    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.config.model_path)

    def predict_future_values(self, model, sensor_id, num_periods=24):
        Current_Date = datetime.today()
        NextDay_Date = datetime.today() + timedelta(days=1)

        # Predict for future dates
        future_dates = pd.date_range(start=Current_Date, end=NextDay_Date, freq='H')
        # future_x[0]= 'Clock'

        future_x = create_features(pd.DataFrame({'sensor': [sensor_id] * len(future_dates)}, index=future_dates))
        future_x['Kwh'] = np.nan
        # Include lag features in future_x
        # print(future_x)
        future_x = add_lags(future_x)
        # print(future_x)'is_holiday',
        FEATURES = ['relative_humidity_2m', 'apparent_temperature',
                    'rain',
                    'lag1', 'lag2', 'lag3', 'day', 'hour', 'month', 'year', 'is_holiday']

        X_all = future_x[FEATURES]

        # Predict future values
        future_predictions = model.predict(X_all)

        # Log future predictions to a CSV file
        future_predictions_df = pd.DataFrame({"predicted_kwh": future_predictions}, index=future_dates)
        future_predictions_file_path = f"future_predictions_sensor_{sensor_id}.csv"
        future_predictions_df.to_csv(future_predictions_file_path)

        mlflow.log_artifact(future_predictions_file_path)

    def log_into_mlflow(self):
        try:
            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # tracking_url_type_store = "http://127.0.0.1:5000"
            print(tracking_url_type_store)

            # client, collection = initialize_mongodb("test")
            # count = collection.count_documents({})
            metrix_dict = {}
            data_files = [file for file in os.listdir(self.config.test_data_path) if file.startswith('test_data_sensor')]
            num = [int(filename.split('_')[-1].split('.')[0]) for filename in data_files]
            num1 = max(num)


            for i in range(0, num1):
                try:
                    test_data = pd.read_csv(os.path.join(self.config.test_data_path, f"test_data_sensor_{i}.csv"))
                except FileNotFoundError:
                    print(f"File not found for test_data_sensor_{i}.csv. Continuing with the next iteration.")
                    continue


                # data_list.append(data_sensor)
                test_data.set_index(['creation_time'], inplace=True, drop=True)
                FEATURES = ['is_holiday',
                            'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                            'precipitation', 'wind_speed_10m', 'wind_speed_100m', 'lag1', 'lag2',
                            'lag3', 'lag4', 'lag5', 'day', 'hour', 'month', 'dayofweek', 'quarter',
                            'dayofyear', 'weekofyear', 'year']
                TARGET = ['consumed_unit']

                test_x = test_data[FEATURES]
                test_y = test_data[TARGET]

                model = loaded_model_dict.get(i)
                # print(model)

                if model is None:
                    logger.warning(f"Model for sensor {i} not found.")
                    continue

                with mlflow.start_run():
                    predicted_kwh = model.predict(test_x)
                    # print(test_y)

                    (rmse, mae, r2) = self.eval_metrics(test_y, predicted_kwh)
                    predicted_df = pd.DataFrame(predicted_kwh, index=test_y.index, columns=["Predicted_Kwh"])

                    # Concatenate the predicted values DataFrame with the actual 'Kwh' values DataFrame
                    result_df = pd.concat([predicted_df, test_y], axis=1)

                    # Print the result

                    # Saving metrics as local
                    scores = {"rmse": rmse, "mae": mae, "r2": r2}
                    print(scores)
                    metrix_dict[f'{i}'] = scores

                    # plt.figure(figsize=(10, 6))
                    # plt.plot(result_df.index, result_df['Predicted_Kwh'], label='Predicted Kwh', color='blue')
                    # plt.plot(result_df.index, result_df['consumed_unit'], label='Actual Kwh', color='red')
                    # plt.xlabel('Time')
                    # plt.ylabel('Kwh')
                    # plt.title('Predicted vs Actual Kwh')
                    # plt.legend()
                    # plt.xticks(rotation=45)
                    # plt.tight_layout()
                    # plt.show()

            save_json(path=Path(self.config.metric_file_name), data=metrix_dict)

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            print(traceback.format_exc())
        # finally:
        #     client.close()
        #     print("db connection closed")
