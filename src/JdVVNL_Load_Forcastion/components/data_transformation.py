import json
import os
import traceback
import warnings
from datetime import datetime, date as datetime_date

import numpy as np
from sklearn.decomposition import PCA

import holidays
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose

from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.entity.config_entity import DataTransformationConfig

warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import threading
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from src.JdVVNL_Load_Forcastion.utils.common import add_lags, create_features, data_from_weather_api, \
    holidays_list, uom, sensor_data


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.le = LabelEncoder()

    # def process_parquet_files(self) -> None:
    #     parquet_files = [file for file in os.listdir(self.config.data_file_path) if file.startswith("circle_id")]
    #     threads = []
    #     for file_name in parquet_files:
    #         thread = threading.Thread(target=self.initiate_data_transformation, args=(os.path.join(self.config.data_file_path, file_name),))
    #         print(thread)
    #         threads.append(thread)
    #         thread.start()
    #
    #     # Wait for all threads to complete
    #     for thread in threads:
    #         thread.join()
    #
    #     print("All files processed.")

    def initiate_data_transformation(self):
        try:
            data_files = [file for file in os.listdir(self.config.data_file_path) if file.startswith("circle_id")]
            endDate = '2024-01-01 00:00:00'
            startDate = '2024-02-26 23:59:59'
            holiday_lst = holidays_list(endDate, startDate)

            for data_file in data_files:
                df1 = pd.read_parquet(os.path.join(self.config.data_file_path, data_file))
                complete_df = df1[['sensor_id', 'creation_time', 'opening_KWh']]
                complete_df = complete_df.round(2)
                complete_df['creation_time'] = pd.to_datetime(complete_df['creation_time'])

                s_data = sensor_data(complete_df['sensor_id'].unique())
                dframe = pd.merge(s_data, complete_df, on='sensor_id', how="inner")
                dframe.set_index('creation_time', inplace=True, drop=True)
                pd.set_option('display.max_columns', 500)
                dframe['label_sensor'] = self.le.fit_transform(dframe['sensor_id'])
                sensor_ids = dframe['label_sensor'].unique()

                # Create a dictionary to map encoded values to original sensor IDs
                self.sensorDecode(sensor_ids)

                outlier_dict = {}
                for s_id, data in dframe.groupby('label_sensor'):
                    s_df = data.copy()
                    print("====================================")

                    print(s_df)

                    if len(s_df) > 3000:  # small dataset
                        description = s_df.describe()
                        Q2 = description.loc['50%', 'opening_KWh']
                        if Q2 < 1:
                            continue

                        # outage situation
                        s_df.loc[s_df['opening_KWh'] == 0, "opening_KWh"] = np.nan
                        s_df.loc[s_df['opening_KWh'].first_valid_index():]
                        s_df.fillna(method="bfill", inplace=True)

                        # missing packet
                        sensor_df = s_df.resample(rule="15min").asfreq()
                        sensor_df.interpolate(method="linear", inplace=True)
                        print("====================================")

                        print(sensor_df)

                        # no consumption / same reading
                        if sensor_df['opening_KWh'].nunique() < 10:
                            continue

                        # previous value of opening_KWh
                        sensor_df['prev_KWh'] = sensor_df['opening_KWh'].shift(1)
                        sensor_df.dropna(inplace=True)
                        if len(sensor_df[sensor_df['prev_KWh'] > sensor_df['opening_KWh']]) > 25:
                            continue

                        # recheck
                        # consumed unit
                        sensor_df['consumed_unit'] = abs(sensor_df['opening_KWh'] - sensor_df['prev_KWh'])
                        if sensor_df['consumed_unit'].nunique() < 10:
                            continue

                        # sensor_df['sensor_id'] = s_id

                        # eliminating id's based on slope
                        numeric_index = pd.to_numeric(sensor_df.index)
                        correlation = np.corrcoef(numeric_index, sensor_df['opening_KWh'])[0, 1]
                        coeffs = np.polyfit(numeric_index, sensor_df['opening_KWh'], 1)
                        slope = coeffs[0]
                        if not np.abs(correlation) > 0.8 and slope > 0:
                            continue

                        # outlier detection
                        description1 = sensor_df.describe()
                        Q1 = description1.loc['25%', 'consumed_unit']
                        Q3 = description1.loc['75%', 'consumed_unit']
                        # IQR
                        u_limit = Q3 + ((Q3 - Q1) * 2)
                        # Z_test
                        z_scores = ((sensor_df['consumed_unit'] - sensor_df['consumed_unit'].mean()) / sensor_df[
                            'consumed_unit'].std())

                        # outliers dataframe rows storing in dictionary to handle outliers
                        outlier_dict['mean'] = sensor_df[
                            sensor_df['consumed_unit'] > sensor_df['consumed_unit'].mean() * 3]
                        outlier_dict['IQR'] = sensor_df[sensor_df['consumed_unit'] > u_limit]
                        outlier_dict['z_score_3'] = sensor_df[abs(z_scores) > 3]
                        outlier_dict['z_score_4'] = sensor_df[abs(z_scores) > 4]

                        # using that method which has the least number of outliers
                        l1 = []
                        for lst in outlier_dict.values():
                            l1.append(len(lst))
                        min_length = min(l1)
                        indices = [key for key, value in outlier_dict.items() if len(value) == min_length]

                        # filling nan on places of outliers and filling the previous value
                        sensor_df.loc[outlier_dict[indices[0]].index, 'consumed_unit'] = np.nan
                        sensor_df.fillna(method="bfill", inplace=True)

                        # adding multiple values from sensor data
                        sensor_df['site_id'] = s_df['site_id']
                        sensor_df.reset_index(inplace=True)
                        print("====================================")

                        print("shuabham")
                        print(sensor_df)
                        # adding weather data and holidays
                        sensor_df['is_holiday'] = sensor_df['creation_time'].dt.date.isin(holiday_lst).astype(int)
                        site = sensor_df['site_id']
                        startDate = sensor_df['creation_time'].min()
                        endDate = sensor_df['creation_time'].max()

                        weather_data = data_from_weather_api(site, startDate, endDate)
                        if not weather_data.empty:
                            # Convert the creation_time columns to datetime if they are not already
                            weather_data['creation_time'] = pd.to_datetime(weather_data['time'])
                            sensor_df['creation_time'] = pd.to_datetime(sensor_df['creation_time'])
                            sensor_df['creation_time_rounded'] = sensor_df['creation_time'].dt.round('H')
                            # Merge sensor_df with weather_data
                            merged_df = pd.merge_asof(sensor_df, weather_data, left_on='creation_time_rounded',
                                                      right_on='creation_time', direction='nearest')
                            print(merged_df.columns)

                            # Drop the redundant columns
                            merged_df.drop(
                                columns=['UOM', 'meter_MWh_mf', 'prev_KWh', 'opening_KWh', 'creation_time_rounded', 'creation_time_y',
                                         'site_id_y', 'site_id_x', 'creation_time_iso'], inplace=True)
                            merged_df.rename(columns={'creation_time_x': 'creation_time'}, inplace=True)
                        else:
                            print("Weather data is empty. Skipping merge operation or performing alternative action...")
                            continue
                        print("====================================")
                        print(merged_df)


                        merged_df.set_index(['creation_time'], inplace=True, drop=True)
                        dfresample = add_lags(merged_df)
                        dfresample = create_features(dfresample)
                        dfresample.reset_index(inplace=True)

                        tss = TimeSeriesSplit(n_splits=5, test_size=24 * 2 * 1, gap=24)

                        for train_idx, val_idx in tss.split(dfresample):
                            train_data = dfresample.iloc[train_idx]
                            test_data = dfresample.iloc[val_idx]

                        train_data_filepath = os.path.join(self.config.root_dir, f"train_data_sensor_{s_id}.csv")
                        test_data_filepath = os.path.join(self.config.root_dir, f"test_data_sensor_{s_id}.csv")
                        data_filepath = os.path.join(self.config.data_file_path, f"sensor_{s_id}.csv")

                        # Save data to separate train and test files for each sensor
                        train_data.to_csv(train_data_filepath, mode='w', header=True, index=False)
                        test_data.to_csv(test_data_filepath, mode='w', header=True, index=False)
                        dfresample.to_csv(data_filepath, mode='w', header=True, index=False)

        except Exception as e:
            print(traceback.format_exc())
            logger.info(f"Error occur in Data Transformation Layer {e}")

    def sensorDecode(self, sensor_ids) -> None:
        try:
            decoded_values = self.le.inverse_transform(sensor_ids)
            encoded_to_sensor_mapping = {str(encoded_value): str(original_sensor_id) for
                                         encoded_value, original_sensor_id in
                                         zip(sensor_ids, decoded_values)}

            # Print the mapping
            print("Encoded to Sensor ID Mapping:")
            for encoded_value, original_sensor_id in encoded_to_sensor_mapping.items():
                print(f"Encoded Value {encoded_value} corresponds to Sensor ID {original_sensor_id}")

            # Write the mapping to a JSON file
            output_file_path = 'encoded_to_sensor_mapping.json'
            with open(output_file_path, 'w') as file:
                json.dump(encoded_to_sensor_mapping, file)

        except ValueError as e:
            # Handle the case where unknown values are encountered
            print(f"Error decoding sensor IDs: {e}")



