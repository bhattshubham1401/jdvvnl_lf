import os
import traceback

import joblib
import pandas as pd

from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.utils.common import create_features, add_lags, store_predictions_in_mongodb, holidays_list, data_from_weather_api


class PredictionPipeline:
    def __init__(self):
        self.path = 'artifacts/data_ingestion/'
        self.model = 'artifacts/model_trainer/model.joblib'

    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.model)

    def predict(self):
        try:
            data_files = [file for file in os.listdir(self.path) if file.startswith('sensor')]

            data_list = []

            # adding holidays
            endDate = '2024-01-01 00:00:00'
            startDate = '2024-03-12 23:59:59'
            holiday_lst = holidays_list(endDate, startDate)


            for data_file in data_files:
                data_sensor = pd.read_csv(os.path.join(self.path, data_file))
                data_list.append(data_sensor)
                weather_data = data_from_weather_api(data_list['site_id'], endDate, startDate)
                # self.actualData(data_sensor)
            # print(data_list)

            # Concatenate data for all sensors
            train_data = pd.concat(data_list, ignore_index=True)
            # print(train_data.tail())

            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

            for sensor_id in train_data['labeled_id'].unique():
                model = loaded_model_dict.get(sensor_id)

                if model is None:
                    logger.warning(f"Model for sensor {sensor_id} not found.")
                    continue

                # Filter data for the current sensor
                # print(sensor_id)
                df = train_data[train_data['labeled_id'] == sensor_id]
                # print("========================================================================================")
                # print(df)
                # print("========================================================================================")
                df.set_index('creation_time', inplace=True)
                # print(df)

                columns_to_drop = ['is_holiday',
                            'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                            'precipitation', 'wind_speed_10m', 'wind_speed_100m', 'lag1', 'lag2',
                            'lag3', 'lag4', 'lag5', 'day', 'hour', 'month', 'dayofweek', 'quarter',
                            'dayofyear', 'weekofyear', 'year']
                columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
                df1 = df.drop(columns_to_drop_existing, axis=1, errors='ignore')
                pd.set_option('display.max_columns', 500)
                # print(df1)

                # index = df1.index.max()
                # endDate = datetime.date.today() + datetime.timedelta(days=7)
                # startDate = datetime.today().strftime('%Y-%m-%d')

                future = pd.date_range(start='2024-03-01 00:00:00', end='2024-04-01 00:00:00', freq='1D')
                future_df = pd.DataFrame(index=future)
                future_df['isFuture'] = True
                df1['isFuture'] = False
                df_and_future = pd.concat([df1, future_df])
                df_and_future = create_features(df_and_future)
                df_and_future = add_lags(df_and_future)
                df_and_future.index.name = 'Clock'
                # df_and_future.reset_index(['Clock'], inplace=True)

                df_and_future = pd.merge(df_and_future, weather_data, on="Clock")
                df_and_future['is_holiday'] = df_and_future['Clock'].dt.date.isin(holiday_lst).astype(int)
                df_and_future.set_index(['Clock'], inplace=True, drop=True)
                df_and_future['sensor'] = sensor_id
                # pd.set_option('display.max_columns', 500)

                future_w_features = df_and_future.query('isFuture').copy()
                # print(sensor_id)
                # print(future_w_features)
                FEATURES = ['is_holiday',
                            'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                            'precipitation', 'wind_speed_10m', 'wind_speed_100m', 'lag1', 'lag2',
                            'lag3', 'lag4', 'lag5', 'day', 'hour', 'month', 'dayofweek', 'quarter',
                            'dayofyear', 'weekofyear', 'year']
                future_w_features['weekofyear'] = future_w_features['weekofyear'].astype(int)
                future_w_features['pred'] = model.predict(future_w_features[FEATURES])
                # print(future_w_features['pred'])
                # future_w_features['pred_shifted'] = future_w_features['pred'].shift(-1)
                # print(sensor_id)
                # return
                store_predictions_in_mongodb(sensor_id, future_w_features.index, future_w_features['pred'])
            return f"Data stored"



        except Exception as e:
            logger.error(f"Error in Model Prediction: {e}")
            print(traceback.format_exc())
