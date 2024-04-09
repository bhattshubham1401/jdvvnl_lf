import os
import traceback
from datetime import datetime, timedelta

import joblib
import pandas as pd

from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.utils.common import create_features, add_lagsV1, store_predictions_in_mongodb, \
    holidays_list, data_from_weather_api, sensor_data


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
            for data_file in data_files:
                data_sensor = pd.read_csv(os.path.join(self.path, data_file))

                sensor_ids = list(data_sensor['sensor_id'].unique())
                s_data = sensor_data(sensor_ids)
                # print(s_data)

                sensor_site_map = dict(zip(s_data['sensor_id'], s_data['site_id']))

                data_sensor['site_id'] = data_sensor['sensor_id'].map(sensor_site_map)
                data_list.append(data_sensor)
                print("Appended data for file:", data_file)

            print("Number of files processed:", len(data_files))
            print("Number of DataFrames appended:", len(data_list))

            # Concatenate data for all sensors
            train_data = pd.concat(data_list, ignore_index=True)
            # print(train_data.tail())

            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

            for sensor_id in train_data['label_sensor'].unique():
                model = loaded_model_dict.get(sensor_id)

                if model is None:
                    logger.warning(f"Model for sensor {sensor_id} not found.")
                    continue

                df = train_data[train_data['label_sensor'] == sensor_id]
                df.set_index('creation_time', inplace=True)
                startDate, future_date, end_date = self.get_date(data_sensor['creation_time'])
                weather_data = data_from_weather_api(df['site_id'], startDate, future_date)
                holiday_lst = holidays_list(startDate, future_date)

                # weather_data.to_csv('weather_data.csv', index=False)

                # print(df)

                columns_to_drop = ['site_id', 'wind_speed_100m', 'lag4', 'lag5', '_id_y', 'temperature_2m',
                                   ' sensor_id ', '_id_x', 'meter_ct_mf', 'asset_id', 'label_sensor', 'precipitation',
                                   'day', 'weekofyear', 'index', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month',
                                   'year', 'lag1', 'lag2', 'wind_speed_10m',
                                   'lag3', 'apparent_temperature', 'rain', 'relative_humidity_2m', 'is_holiday',
                                   ' precipitation', 'cloud_cover']

                columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
                df1 = df.drop(columns_to_drop_existing, axis=1, errors='ignore')
                pd.set_option('display.max_columns', 500)
                # print(df1)

                # index = df1.index.max()
                # endDate = datetime.date.today() + datetime.timedelta(days=7)
                # startDate = datetime.today().strftime('%Y-%m-%d')

                future = pd.date_range(end_date, future_date, freq='15 min')
                future_df = pd.DataFrame(index=future)
                future_df['isFuture'] = True
                df1['isFuture'] = False
                df_and_future = pd.concat([df1, future_df])

                df_and_future.index.name = 'creation_time'
                df_and_future.reset_index(['creation_time'], inplace=True)
                # df_and_future.to_csv('df_and_future.csv', index=False)
                df_and_future['creation_time'] = pd.to_datetime(df_and_future['creation_time'])

                # weather_data = data_from_weather_api(site, startDate, endDate)
                if not weather_data.empty:
                    df_and_future['creation_time_rounded'] = pd.to_datetime(df_and_future['creation_time']).dt.round(
                        'H')
                    weather_data['creation_time_rounded'] = pd.to_datetime(weather_data['time']).dt.round('H')
                    merged_df = pd.merge(df_and_future, weather_data, on='creation_time_rounded', how='left')
                    merged_df.drop(columns=['creation_time_rounded'], inplace=True)
                    merged_df.reset_index(drop=True, inplace=True)

                # Adding holidays
                merged_df['is_holiday'] = merged_df['creation_time'].dt.date.isin(holiday_lst).astype(int)
                # merged_df.to_csv('merged_df_before.csv', index=False)
                merged_df.set_index(['creation_time'], inplace=True, drop=True)
                merged_df = add_lagsV1(merged_df)
                # print(merged_df.columns)

                merged_df = create_features(merged_df)
                merged_df['weekofyear'] = merged_df['weekofyear'].astype(int)
                # merged_df.reset_index(['creation_time'], inplace=True)
                # merged_df.to_csv('merged_df.csv', index=False)

                future_w_features = merged_df.query('isFuture').copy()
                FEATURES = ['is_holiday',
                            'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                            'precipitation', 'wind_speed_10m', 'wind_speed_100m', 'lag1', 'lag2',
                            'lag3', 'lag4', 'lag5', 'day', 'hour', 'month', 'dayofweek', 'quarter',
                            'dayofyear', 'weekofyear', 'year']
                # print(future_w_features[FEATURES].info())

                future_w_features['pred'] = model.predict(future_w_features[FEATURES])
                print(future_w_features)
                # future_w_features['pred'].to_csv('merged_df.csv', index=False)
                store_predictions_in_mongodb(sensor_id, future_w_features.index, future_w_features['pred'])

        except Exception as e:
            logger.error(f"Error in Model Prediction: {e}")
            print(traceback.format_exc())

    def get_date(self, startDate_str):
        start_date = startDate_str.min()

        end_date = startDate_str.max()

        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        startDate = start_date.replace(hour=0, minute=0, second=0)

        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        future_date = end_date + timedelta(days=45)
        future_date = future_date.replace(hour=23, minute=59, second=59)
        return startDate, future_date, end_date
