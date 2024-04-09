import os
import traceback

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.JdVVNL_Load_Forcastion import logger
from src.JdVVNL_Load_Forcastion.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            data_files = [file for file in os.listdir(self.config.train_data_path) if file.startswith('train')]
            num = [int(filename.split('_')[-1].split('.')[0]) for filename in data_files]
            num1 = max(num)
            models_dict = {}

            for i in range(0, num1):
                try:
                    df = pd.read_csv(os.path.join(self.config.train_data_path, f"train_data_sensor_{i}.csv"))
                except FileNotFoundError:
                    print(f"File not found for train_data_sensor_{i}.csv. Continuing with the next iteration.")
                    continue

                # y_train = pd.read_csv(os.path.join(self.config.train_data_path, f"test_data_sensor_{i}.csv"))
                df.set_index('creation_time', inplace=True)

                FEATURES = ['is_holiday',
                            'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                            'precipitation', 'wind_speed_10m', 'wind_speed_100m', 'lag1', 'lag2',
                            'lag3', 'lag4', 'lag5', 'day', 'hour', 'month', 'dayofweek', 'quarter',
                            'dayofyear', 'weekofyear', 'year']
                TARGET = ['consumed_unit']

                X_train = df[FEATURES]
                y_train = df[TARGET]

                # Train an XGBoost model on the sensor's data
                xgb_model = XGBRegressor()
                xgb_model.fit(X_train, y_train)

                # Calculate and print model evaluation metrics for this sensor
                train_score = xgb_model.score(X_train, y_train)
                print(f"Train Score for sensor {i}: {train_score}")

                # Perform hyperparameter tuning using RandomizedSearchCV
                param_grid = {
                    'n_estimators': self.config.n_estimators,
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'subsample': self.config.subsample,
                    'eta': self.config.subsample,
                    'colsample_bytree': self.config.colsample_bytree,
                    'reg_alpha': self.config.reg_alpha
                }

                random_search = RandomizedSearchCV(xgb_model,
                                                   param_distributions=param_grid,
                                                   n_iter=10,
                                                   scoring='neg_mean_squared_error',
                                                   cv=5,
                                                   # verbose=1,
                                                   n_jobs=-1,
                                                   random_state=45)

                # Fit the RandomizedSearchCV to the data
                random_search.fit(X_train, y_train)

                # Get the best parameters
                best_params = random_search.best_params_
                print(f"Best Parameters for sensor {i}: {best_params}")

                # Train the model with the best parameters
                best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              learning_rate=best_params['learning_rate'],
                                              subsample=best_params['subsample'],
                                              # eta=best_params['eta'],
                                              colsample_bytree=best_params['colsample_bytree'],
                                              reg_alpha=best_params['reg_alpha'],
                                              reg_lambda=0.01,
                                              )
                best_xgb_model.fit(X_train, y_train)

                models_dict[i] = best_xgb_model
                # print(models_dict)
            # Save the dictionary of models as a single .joblib file
            joblib.dump(models_dict, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")
