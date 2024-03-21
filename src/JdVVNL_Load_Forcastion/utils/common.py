'''functionality that we are use in our code'''

import json
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import holidays
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from dotenv import load_dotenv
from ensure import ensure_annotations
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

from src.JdVVNL_Load_Forcastion import logger

# from src.mlProject.components.data_transformation import sensorDecode
# from statsmodels.tsa.stattools import adfuller

load_dotenv()


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args\\\\\\\\\\\\\\
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
       memory_usage = hourly_data.memory_usage(deep=True).sum() / (1024 ** 2)
    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def convert_datetime(input_datetime):
    parsed_datetime = datetime.strptime(input_datetime, '%Y-%m-%dT%H:%M')
    formatted_datetime = parsed_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_datetime


# @ensure_annotations
# def get_mongoData():
#     ''' calling DB configuration '''
#
#     logger.info("calling DB configuration")
#     db = os.getenv("db")
#     host = os.getenv("host")
#     port = os.getenv("port")
#     collection = os.getenv("collection")
#
#     MONGO_URL = f"mongodb://{host}:{port}"
#
#     ''' Read data from DB'''
#
#     '''Writing logs'''
#     logger.info("Reading data from Mongo DB")
#
#     '''Exception Handling'''
#
#     try:
#         client = MongoClient(MONGO_URL)
#         db1 = client[db]
#         collection = db1[collection]
#
#         data = collection.find({})
#
#         columns = ['sensor', 'Clock', 'R_Voltage', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current',
#                    'B_Current', 'A', 'BlockEnergy-WhExp', 'B', 'C', 'D', 'BlockEnergy-VAhExp',
#                    'Kwh', 'BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp']
#
#         datalist = [(entry['sensor_id'], entry['raw_data']) for entry in data]
#         df = pd.DataFrame([row[0].split(',') + row[1].split(',') for row in datalist], columns=columns)
#
#         '''Dropping Columns'''
#         df = df.drop(
#             ['BlockEnergy-WhExp', 'A', 'B', 'C', 'D', 'BlockEnergy-VAhExp', 'BlockEnergy-VAhExp', 'BlockEnergy-VArhQ1',
#              'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp'], axis=1)
#         pd.set_option('display.max_columns', None)
#
#         # print("===============DataType Conversion==================")
#         df['Clock'] = pd.to_datetime(df['Clock'])
#         df['Kwh'] = df['Kwh'].astype(float)
#         df['R_Voltage'] = df['R_Voltage'].astype(float)
#         df['Y_Voltage'] = df['Y_Voltage'].astype(float)
#         df['B_Voltage'] = df['B_Voltage'].astype(float)
#         df['R_Current'] = df['R_Current'].astype(float)
#         df['Y_Current'] = df['Y_Current'].astype(float)
#         df['B_Current'] = df['B_Current'].astype(float)
#         return df
#
#     except Exception as e:
#         logger.info(f"Error occurs =========== {e}")

''' Fetching Data from mongo DB through API'''


@ensure_annotations
def get_data_from_api_query():
    ''' '''
    try:
        # client = pymongo.MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB
        # db = client["your_database_name"]  # Specify the database name
        # collection = db["dlms_load_profile"]  # Specify the collection name
        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection = os.getenv("collection1")

        MONGO_URL = f"mongodb://{host}:{port}"

        ''' Read data from DB'''

        '''Writing logs'''
        logger.info("Reading data from Mongo DB")

        '''Exception Handling'''
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection]
        lst = ['0039924a-0c2f-434f-90a2-6ecaec53f14b',
               '00558534-4976-425b-a0a3-72af70d2ef28',
               '00f1263a-1373-49ee-aabe-77f22ddd21b9',
               '0175d32f-3439-46ae-b037-ae8139daec97',
               '02c42b8d-c042-4e5d-bf2e-80d8335e8598',
               '03339b4c-83cd-44c8-be39-33fc492663f7',
               '060a496f-492d-4d55-8029-cf74d449191c',
               '075f28dd-cc46-4590-ae24-5cb2013a75d3',
               '07dbe951-aef5-4288-919e-734a3556e21c',
               '08369f94-0c73-4993-abe2-449ec8222593',
               '086faca4-5c84-4f9b-899d-2548ad60907f',
               '08dee2b7-a111-45c4-9cbc-15a1c14ea3db',
               '098b52a6-1af6-4342-a913-6e7ab7f22146',
               '0a8b4f20-1cdc-4445-a738-ac5c4836c5f1',
               '0b244a8f-efa6-4de0-93f4-6649b5e05c60',
               '0bbf5a51-3804-4a57-88c9-73540398631d',
               '0c7d9ccf-9bde-4785-8742-50f5f90a720c',
               '0d0578d5-f7ff-4d06-aba0-a1e2c30b6608',
               '0dfed391-7193-4c90-9356-9cfd6e30cd59',
               '0e163bd1-0a56-4ccd-8558-4f8d1f998ea1',
               '0ea4e693-c63a-462c-a9ff-ddd6db07d076',
               '0f2db363-b2e1-471e-9a68-fab0afb36d66',
               '105eebc7-e8a5-4701-a58f-9e0960ac1cf7',
               '1069272d-ff0f-49cb-ab97-73f71d27c2cd',
               '113f0385-8700-433e-ad16-b0994e11f9d0',
               '13bd19d7-7f29-4f7c-807f-5245402d2ac8',
               '1460a75e-c5e6-4666-81d6-ca41486caaf3',
               '165c1edb-a884-4421-8440-88200715b5da',
               '16e65d6e-e924-4bcd-b91e-e542c76dad14',
               '17f0fd7b-8b73-42e8-b373-6830d1512196',
               '1872d793-df7a-48b9-be54-91174004c4cc',
               '1911709f-7aa5-4f88-b18c-6f13055f36aa',
               '1b27c01d-14c0-4ef5-8111-d854be607ca1',
               '1b51a507-2dce-46f2-aac8-1c426b1c015a',
               '1db1fb04-0ffc-4ccc-9a66-3927329ab7c6',
               '1fa2edc4-96d1-42c1-8684-45cdb4944158',
               '2117ec15-a4d3-4f08-845a-b70a9b4f0e2d',
               '21fed0d0-8415-49f6-8971-43c9b5fc9613',
               '24367908-26c6-45e4-bacc-6101c2b69e67',
               '2469674f-1c57-4680-89d6-fff46e2b8c9f',
               '246ec061-be2e-4904-8ebb-1b2c5da5db63',
               '256dec71-3610-42dc-8af1-812da1c3efa8',
               '25fae914-10a8-4e96-b80e-3be629470a0d',
               '2967fa2f-4ec9-47f7-84e4-2f8630fd6fbf',
               '2a2e95ec-7690-4c70-8497-7d0bf2575a67',
               '2bdfdd2d-4beb-4e6b-b2a9-67968755509c',
               '2c129d1c-b8a2-4cbe-862e-3b87fda62eb4',
               '2c40659d-be11-4e75-9d89-020798f464f3',
               '2e11b38b-dbd3-4ba6-b67b-bd553ef4a266',
               '2ef71e5e-2227-45cf-9e1f-5c5dd389dd69',
               '2f961e1d-6dd2-4314-9c0f-04829caa6b6d',
               '2fc85193-6e31-467c-910f-c3c60bdb623e',
               '31c821cb-cf5a-4422-a3d7-19cf4cbcdb3d',
               '32094073-f000-4bd9-b99f-0e8685ed1d2c',
               '322df352-3e75-4049-8195-e79a04d9adc3',
               '35752a44-c9d1-40ec-8264-cc736d90179d',
               '363f7c9f-893c-4bf8-9db4-fdc2f3e0f3c2',
               '36cc76b5-75bc-4396-b18d-96ac432bb77b',
               '3776209d-82c4-4a8f-bca2-db799731b979',
               '38ba013f-5ece-4be9-baf5-3db54f747543',
               '392be209-fc56-4c8c-9948-f79eb20cd49e',
               '3a236028-4b67-4622-b630-0d2f628d7e4e',
               '3ab2e68e-9080-4bc0-ae4a-2ae366682ba3',
               '3aebe108-7e31-4596-aa26-618731febc1f',
               '3d95f549-9afc-4e6a-b221-001eb6279a34',
               '3df95dde-7ed4-45a6-876e-d206f40284a4',
               '3f503c6b-e882-478b-8860-f64cd566e52c',
               '4173e084-ccf3-47ed-bc97-e0961451a168',
               '4213b14b-96b9-4c67-8316-7fb0f641d2bb',
               '455120f5-66ad-4173-87f1-e0b44b38d976',
               '466257e2-71ea-4a85-8913-7c61fc20af0d',
               '4686eda2-5852-416a-bf5e-e690d14c667f',
               '472bd83d-3a4f-416f-897e-1e8227dc9149',
               '49447e2b-d18a-4c76-bfb6-11a0c0a665aa',
               '4bba407e-e653-4cdd-923b-3e1b160095f3',
               '4d444eea-e557-4a81-9197-212c5bc31855',
               '4e3dacc6-bf6d-48d9-b05b-a2abf322a04c',
               '4eb1da12-b310-4a9e-8c5d-7af629aad5f3',
               '4f64c748-074a-4c96-b823-a753542b25dd',
               '521cdf72-ed97-4f64-86cd-b9d351d9ec64',
               '521eba00-b0a6-4b20-9486-275791f7242c',
               '5276a8a1-9351-47cb-8ea9-2db61512ae31',
               '5320e59b-f178-468c-abf5-1805facefaec',
               '5685bebd-f782-4e83-875f-fd1da595781c',
               '5697b8ca-faea-4082-8b04-ff8dcdf2be3c',
               '578c40fc-1e26-4f01-971e-8094b16cc928',
               '5939ce4c-e08a-4ca9-8c57-1e0025577f08',
               '59de93c2-e1d8-490e-b2de-4ec028a5fb42',
               '5b575f93-e408-4317-8345-1fb1ba395554',
               '5ba5b890-420f-455b-97c1-51b18fc86b7d',
               '5eccce4c-a64e-44b0-af69-1a5d5fa012a2',
               '609889a4-b6f4-44fc-a3b1-9212aa5b81f6',
               '6102a4f9-8b2e-4b51-897c-68c44c60f9f4',
               '66b3499c-a9b6-41c5-835f-02b66b89f1d3',
               '6916bd3c-1898-4b92-87d8-aa4c2ca17b63',
               '69eb446a-df24-4546-b381-d44dcc991916',
               '6a66bdf3-ae70-45f5-a241-a73927b0f14d',
               '6ad5d4b4-e44f-40a6-9951-03552fda0efa',
               '6c7093e6-cf7f-4d65-9674-346a9c16c1b1',
               '6cd70ab6-35e3-47a5-9340-8d6fb7f01ee7',
               '6db0be53-dd45-479d-8342-6102b9ceab61',
               '7219488e-cac5-46dd-a1a5-117ae27e8530',
               '725cb4ed-0f69-41a4-ac75-757e58cf54c1',
               '7271aeb1-df82-44ae-b647-907c53c3fe86',
               '72ed5f97-6a00-437a-8f08-5acaab9fd7f2',
               '73364069-56cc-42a3-8b5b-0e54e908e51e',
               '738a7b0d-5180-4117-931d-602e0f932697',
               '758a7244-3ad2-4249-b91f-33a3e43cb9be',
               '758bb47e-81e2-411c-b57e-eaf9296b8e1d',
               '75e7203f-013e-4e61-a8f9-5205face3960',
               '77601b36-2e9d-4702-adda-ca7a3d3638c9',
               '77d8f029-d3bc-48b1-b199-442bbbc3e929',
               '7b02bf1e-afbe-43dc-9e93-77bcf4073f35',
               '7d17d4ba-bb8c-4727-89e0-07b43e27af5b',
               '7dbf6e44-fcc4-431a-b530-e953426af7d6',
               '7f116eed-4125-4709-92b5-422539a4aad2',
               '81e9bdfb-f5f1-43db-83e7-eb663e509546',
               '824b69eb-8071-497c-9f80-2574aace38fc',
               '85b62714-4f82-40d1-a900-d6eaebc0f63f',
               '87ac5fcb-9b2c-40da-8a32-c18053fbb6ea',
               '88deb37c-56f0-4845-a951-c9ca5aa1643c',
               '8a97a5db-2779-4f25-95b4-db4e1abcd461',
               '8b580343-c841-4c99-992d-76e0ed9e12f1',
               '8c77187c-4df9-4f59-8c4c-f6f1eecc4710',
               '8cc8b0b7-561f-477f-9f8b-e853ebb64014',
               '8d7dca0a-64b0-4bfe-9230-da9ca85649a6',
               '90a3b84b-9911-4dc4-a9d0-6f9d55f26e13',
               '9237e85c-a6d8-4c8c-9c08-a115cc93a3a2',
               '92df1179-1670-4393-9c4b-119f1239c45d',
               '94343aff-c205-49f5-8841-5e3a0de6db27',
               '94571f80-4928-4023-9cb6-6d61c7cb00f7',
               '9546a5ef-da12-4278-a21a-f667a85d4346',
               '96dbe2b3-8b10-481d-a719-efcea49dceb3',
               '9775efb3-10f3-4dd8-8bf4-1dabac60bb11',
               '99919fd7-495f-47fd-b0b3-5c85fc635387',
               '9bc735c1-6dd5-4e3d-a080-4da479d3c344',
               '9d2c6c67-3459-4eff-be8d-3ddd2b54d54b',
               '9da00b94-49e8-4e65-8299-1eb840281890',
               '9f9f4f5c-0cb9-4d31-a34b-847d286613f8',
               'a182bb4f-5021-448a-b564-7d625ce79c8b',
               'a2c6ea94-eec4-47b0-b9af-0d914f9ce13e',
               'a2fc1700-2cfd-437b-af10-343a24b0525d',
               'a50085e1-004c-453a-9130-3f1335caa5ff',
               'ab4bad73-6364-4b3d-9179-1ff5418d13c2',
               'ab4db663-c446-49ae-9e04-ea3a3d68e1de',
               'ab587ba9-dbe7-433e-87ae-709264435ed3',
               'ad293888-d47e-463f-a9ad-f8e9c3c5618a',
               'ae39bcc3-41eb-4a84-ae09-8a4f7598baeb',
               'aeeeeb00-aea8-4ee5-b907-5cfff2bc9e5d',
               'afc27336-5df2-41d5-9b4d-b9e1dbce872a',
               'b3cedabe-cdcb-4fa0-84b6-105ec4d79101',
               'b3ef3afc-884c-4cb8-a5fc-8cfb27130095',
               'b4286061-c0f6-4939-9d87-6159fd8c325e',
               'b90c8527-ff54-4272-9092-0b45b983d812',
               'bebfc66c-e868-4437-a551-49d26f59aed0',
               'bfc0e2ed-4c55-4edf-9666-7d0d2006a4f5',
               'c03b9195-29e4-4269-9ddc-87134b4f8f24',
               'c10870fe-0759-4114-b044-75a328c70b08',
               'c3da6350-3f0e-42a3-b3cd-ecd33020e2b5',
               'c7234493-9933-4dd9-9523-56ed01e268eb',
               'c93e8c35-c20a-42a8-b89d-05a177f2acd2',
               'c96340c2-fe77-48ec-bbce-54b8b4fc2964',
               'ca2ee423-150a-4a95-97a4-958ec33f5385',
               'ca357306-6c69-4315-a043-568664ab1a70',
               'cc8488c5-7f24-4ca4-b17d-69e20c5a63bc',
               'ce4834cc-b1b6-4db3-a82c-3de5c0171d46',
               'd009d687-cfff-4a26-84d0-d098d44b323f',
               'd113bbf3-5183-4f93-9ec4-1e034bfdb46c',
               'd56adbb2-f5d8-4cf8-9483-5fbcfcfef8a9',
               'd7dd5cbc-89d3-4f6c-9a55-8dc6ef53bd5d',
               'da4adbea-2e30-4cb0-a5dc-d77d238b6b7f',
               'da4de033-b20c-41d7-9a5b-803314484643',
               'da51f7d5-eb0c-4da4-90ad-a7e85cc2fca1',
               'dc7245ce-c078-40d1-9c37-560db8b20d8f',
               'dd005823-3d92-4d54-ac11-e5364673f229',
               'dd9f69dc-23c5-41f9-b443-ae0cd7f97734',
               'df40c133-7076-46c2-b978-3eee41a32a06',
               'e649f4ff-9a30-4941-a70d-97aa0b381a60',
               'e8bbcd26-8085-41a8-b583-58fe1761e71a',
               'ea01de2b-8e63-48e2-80f9-470264a97efa',
               'ea4f095b-9d73-487d-8fa8-2a219f5f2389',
               'ec55b62a-f87d-4d7d-b8cf-0cd8ae7eaefa',
               'ee26c4e1-f7da-4098-803c-a9da2493c883',
               'eeb1ea1e-279a-4afc-8478-b93a76cac83f',
               'f019e271-2cd3-4f7c-a0c4-d2d55f19a5b0',
               'f11beb0b-61ca-46c9-8d72-6b3428c645c6',
               'f18b8345-699e-48ec-b77a-9bfaaea1b391',
               'f1c032da-d55e-4b47-a74f-4bcf09715f02',
               'f34851f4-ec0c-481e-8567-7e3bbfa6953e',
               'f35c8e81-90c5-47ad-a887-2f188493701a',
               'f45904e1-aa5b-4451-ad45-e86d31f7b510',
               'fa6ca047-b654-4156-af80-3df0b01809b5',
               'fbba699f-b4ac-4ba9-aacd-4ae0ffc24e8b',
               'fd476d6e-a227-4a79-a364-ff55032a1bf4',
               'fdf9be19-2c01-4536-a1b4-501b8fce687d',
               'fe9acf54-29b0-4aa9-9352-bbca0764d4f2']
        x = collection.find({"parent_sensor_id": {"$in": lst}})

        data_list = list(x)
        df = pd.DataFrame(data_list, columns=[
            "meter_date", 'parent_sensor_id', 'opening_KWh', 'closing_KWh',
            'consumed_KWh', 'project_id', 'uom', 'cumm_PF', 'creation_time'
        ])

        return df
    except Exception as e:
        logger.info(f"Error occurs =========== {e}")


@ensure_annotations
def load_file():
    file = os.getenv("filename")
    return file


@ensure_annotations
def plotData(df1):
    plt.figure(figsize=(10, 6))
    plt.scatter(df1['KWh'], df1['cumm_PF'], label='Actual')
    plt.xlabel('KWh ')
    plt.ylabel('cumm_PF')
    plt.legend()
    plt.show()
    return

    # Line plot
    # sns.lineplot(x='x_column', y='y_column', data=data)
    # plt.show()
    #
    # # Histogram
    # sns.histplot(data['numeric_column'], bins=10)
    # plt.show()
    #
    # # Box plot
    # sns.boxplot(x='category_column', y='numeric_column', data=data)
    # plt.show()
    #
    # # Bar plot
    # sns.barplot(x='category_column', y='numeric_column', data=data)
    # plt.show()
    #
    # # Pair plot (for exploring relationships between multiple variables)
    # sns.pairplot(data)
    # plt.show()


@ensure_annotations
def sliderPlot(df1):
    fig = px.line(df1, x=df1['meter_timestamp'], y=df1['KWh'])
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )

    )
    fig.show()
    return


@ensure_annotations
# def store_predictions_in_mongodb(sensor_id, dates, predictions):
#     try:
#         logger.info("Calling DB configuration")
#
#         # Load the labeled_to_original_mapping from a JSON file
#         with open("encoded_to_sensor_mapping.json", "r") as file:
#             labeled_to_original_mapping = json.load(file)
#
#         db = os.getenv("db")
#         host = os.getenv("host")
#         port = os.getenv("port")
#         collection_name = os.getenv("collection")
#
#         mongo_url = f"mongodb://{host}:{port}"
#         client = MongoClient(mongo_url)
#         db1 = client[db]
#         collection = db1[collection_name]
#
#         unique_dates = sorted(set(dates.date))
#
#         for date in unique_dates:
#             date_str = date.strftime('%Y-%m-%d')
#             original_sensor_id = labeled_to_original_mapping.get(str(sensor_id), str(sensor_id))
#             document_id = f"{original_sensor_id}_{date_str}"
#
#             data = {
#                 "_id": document_id,
#                 "sensor_id": original_sensor_id,
#                 "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 "day": date.strftime('%d'),
#                 "month": date.strftime('%m'),
#                 "year": date.strftime('%Y'),
#                 "millisecond": int(datetime.now().timestamp() * 1000),
#                 "data": {}
#             }
#
#             # Filter predictions for the current date
#             date_predictions = predictions[dates.date == date]
#
#             # Populate the 'data' dictionary with hourly predictions
#             for i, prediction in enumerate(date_predictions):
#                 prediction_float = round(float(prediction), 4)
#                 data["data"][str(i)] = {
#                     "pre_kwh": prediction_float,
#                     "pre_current": 0.0,
#                     "pre_load": 0.0,
#                     "act_kwh": 0.0,
#                     "act_load": 0.0
#                 }
#
#             data_dict = {key: float(value) if isinstance(value, (float, np.integer, np.floating)) else value
#                          for key, value in data.items()}
#
#             # Insert data into MongoDB
#             collection.insert_one(data_dict)
#
#         client.close()
#         logger.info("Data stored successfully")
#         return
#
#     except Exception as e:
#         print(e)

def store_predictions_in_mongodb(sensor_id, dates, predictions):
    try:
        print(sensor_id)
        print(type(sensor_id))
        sensor_id = round(sensor_id)
        print(type(sensor_id))
        logger.info("Calling DB configuration")

        # Load the labeled_to_original_mapping from a JSON file
        with open("encoded_to_sensor_mapping.json", "r") as file:
            labeled_to_original_mapping = json.load(file)

        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection4")

        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection = db1[collection_name]

        unique_dates = sorted(set(dates.date))

        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            original_sensor_id = labeled_to_original_mapping.get(str(sensor_id), str(sensor_id))
            print(original_sensor_id)
            # return
            document_id = f"{original_sensor_id}_{date_str}"

            data = {
                "_id": document_id,
                "sensor_id": original_sensor_id,
                "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "day": date.strftime('%d'),
                "month": date.strftime('%m'),
                "year": date.strftime('%Y'),
                "millisecond": int(datetime.now().timestamp() * 1000),
                "data": {}
            }

            # Filter predictions for the current date
            date_predictions = list(predictions[dates.date == date])

            # Populate the 'data' dictionary with hourly predictions
            for i, prediction in enumerate(date_predictions):
                prediction_float = round(float(prediction), 4)
                data["data"][str(i)] = {
                    "pre_kwh": float(prediction_float),
                    "pre_current": 0.0,
                    "pre_load": 0.0,
                    "act_kwh": 0.0,
                    "act_load": 0.0
                }

            # Insert data into MongoDB
            collection.insert_one(data)

        client.close()
        logger.info("Data stored successfully")
        return

    except Exception as e:
        logger.error(f"Error in storing data to MongoDB: {e}")
        print(traceback.format_exc())


def create_features(hourly_data):
    hourly_data = hourly_data.copy()

    # Check if the index is in datetime format
    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        hourly_data.index = pd.to_datetime(hourly_data.index)

    hourly_data['day'] = hourly_data.index.day
    hourly_data['hour'] = hourly_data.index.hour
    hourly_data['month'] = hourly_data.index.month
    hourly_data['dayofweek'] = hourly_data.index.dayofweek
    hourly_data['quarter'] = hourly_data.index.quarter
    hourly_data['dayofyear'] = hourly_data.index.dayofyear
    hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
    hourly_data['year'] = hourly_data.index.year
    return hourly_data


@ensure_annotations
def add_lags(df):
    try:
        target_map = df['consumed_unit'].to_dict()
        # 15 minutes, 30 minutes, 1 hour
        df['lag1'] = (df.index - pd.Timedelta('15 minutes')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('30 minutes')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('1 day')).map(target_map)
        df['lag4'] = (df.index - pd.Timedelta('7 days')).map(target_map)
        df['lag5'] = (df.index - pd.Timedelta('15 days')).map(target_map)
    except KeyError as e:
        print(f"Error: {e}. 'consumed_unit' column not found in the DataFrame.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    return df


@ensure_annotations
def rolling_statistics(self, data):
    MA = data.rolling(window=24).mean()
    MSTD = data.rolling(window=24).std()
    plt.figure(figsize=(15, 5))
    orig = plt.plot(data, color='black', label='Original')
    mean = plt.plot(MA, color='red', label='MA')
    std = plt.plot(MSTD, color='yellow', label='MSTD')
    plt.legend(loc='best')
    plt.title("Rolling Mean and standard Deviation")
    plt.show()


@ensure_annotations
def adfuller_test(self, data, sensor_id):
    print("Result od adifuller test:")
    # dftest = adfuller(data, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Stat', 'p-value', 'lags used', 'np of observation used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Values(%s)' % key] = value
    # print(f"The sensor id {sensor_id}-{dfoutput}")


def store_train_test_data_in_db(sensor_id, df):
    try:
        logger.info("calling DB configuration for data")
        host = os.getenv("host")
        port = os.getenv("port")
        db = os.getenv("db")
        collection_name = os.getenv("collection2")

        mongo_url = f"mongodb://{host}:{port}"
        print(mongo_url)
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection = db1[collection_name]
        data1 = df.to_dict(orient='records')
        id_data = {"sensor_id": sensor_id, "data": data1}

        collection.insert_one(id_data)

        client.close()
        logger.info("DB client close")


    except Exception as e:
        print(e)


@ensure_annotations
def data_from_weather_api(site, startDate, endDate):
    ''' Fetch weather data from CSV file based on date range'''
    logger.info("Weather data fetching")
    try:
        start_date = startDate.strftime('%Y-%m-%d %H:%M:%S')
        end_date = endDate.strftime('%Y-%m-%d %H:%M:%S')
        site_array = np.array(site, dtype=object)
        site = str(site_array[0])
        print("Start Date:", start_date)
        print("End Date:", end_date)
        print("Site ID:", site)

        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection3")
        print("Collection:", collection_name)

        MONGO_URL = f"mongodb://{host}:{port}"

        ''' Read data from DB'''

        '''Writing logs'''
        logger.info("Reading data from Mongo DB")

        '''Exception Handling'''
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]
        documents = []
        query = collection.find({
            "time": {
                "$gte": start_date,
                "$lte": end_date
            },
            "site_id": site
        })
        for doc in query:
            documents.append(doc)
            # print(documents)
        try:

            df = pd.DataFrame(documents)
            return df
        except Exception as e:
            print(e)
    except Exception as e:
        print("Error:", e)


@ensure_annotations
# def holidays_list():
#     logger.info("holidays list")
#     try:
#         start_date = datetime_date(2023, 1, 1)
#         end_date = datetime_date(2023, 12, 31)
#         holiday_list = []
#
#         def is_holiday(single_date):
#             year = single_date.year
#             country_holidays = holidays.CountryHoliday('India', years=year)
#             return single_date in country_holidays
#
#         date_list = [
#             (single_date, holidays.CountryHoliday('India', years=single_date.year)[single_date])
#             for single_date in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1))
#             if is_holiday(single_date)]
#         for date, name in date_list:
#             # print(f"{date}: {name}")
#             holiday_list.append(date)
#
#         return holiday_list
#     except Exception as e:
#             print(e)
# def holidays_list(end_date_str, start_date_str):
#     logger.info("Holidays list")
#     try:
#         start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
#         end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
#         print(start_date)
#         print(end_date)
#         # start_date = datetime_date(2023, 1, 1)
#         # end_date = datetime_date(2023, 12, 31)
#         holiday_list = []
#
#         # Get the holiday dates in India for the specified year
#         india_holidays = holidays.CountryHoliday('India', years=start_date.year)
#
#         # Iterate through each date from start_date to end_date
#         current_date = start_date
#         while current_date <= end_date:
#             # Check if the current date is a holiday in India or a Sunday
#             if current_date in india_holidays or current_date.weekday() == 6:
#                 holiday_list.append(current_date)
#             current_date += timedelta(days=1)
#
#         return holiday_list
#
#     except Exception as e:
#         print(e)

@ensure_annotations
# def holidays_list(end_date_str, start_date_str):
#     logger.info("Holidays list")
#     try:
#         print("")
#         end_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
#         start_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
#         holiday_list = []
#
#         # Get the holiday dates in India for the specified year
#         india_holidays = holidays.CountryHoliday('India', years=start_date.year)
#
#         # Iterate through each date from start_date to end_date
#         current_date = start_date
#         while current_date <= end_date:
#             # Check if the current date is a holiday in India or a Sunday
#             if current_date in india_holidays or current_date.weekday() == 6:
#                 holiday_list.append(current_date)
#             current_date += timedelta(days=1)
#
#         return holiday_list
#
#     except Exception as e:
#         print(e)

def holidays_list(start_date_str, end_date_str):
    logger.info("Generating holidays list")
    try:
        # start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        # end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        start_date = start_date_str.date()
        end_date = end_date_str.date()

        holiday_list = []

        # Get the holiday dates in India for the specified year
        india_holidays = holidays.CountryHoliday('India', years=start_date.year)

        # Iterate through each date from start_date to end_date
        current_date = start_date
        while current_date <= end_date:
            # Check if the current date is a holiday in India or a Sunday
            if current_date in india_holidays or current_date.weekday() == 6:
                holiday_list.append(current_date)
            current_date += timedelta(days=1)

        return holiday_list

    except Exception as e:
        logger.error(f"Error in holidays_list: {e}")
        return None


def dataset_count():
    try:
        logger.info("calling DB configuration for data")
        host = os.getenv("host")
        port = os.getenv("port")
        db = os.getenv("db")
        collection_name = os.getenv("collection2")

        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection = db1[collection_name]
        count = collection.count_documents({})
        client.close()

        return count

    except Exception as e:
        print(e)


def data_for_test_train_split():
    try:
        logger.info("calling DB configuration for data")
        host = os.getenv("host")
        port = os.getenv("port")
        db = os.getenv("db")
        collection_name = os.getenv("collection2")

        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection = db1[collection_name]
        client.close()
        logger.info("DB client close")

    except Exception as e:
        print(e)


def data_from_weather(start_date, end_date):
    ''' weather data'''
    logger.info("weather data fetching")
    try:
        start_date1 = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_date1 = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        current_year = start_date1.year
        s_date = start_date1.replace(year=current_year - 1)
        e_date = end_date1.replace(year=current_year - 1)
        df = pd.read_csv("weather1.csv")
        df['Clock'] = pd.to_datetime(df['Clock'])

        newdf = (df['Clock'] > s_date) & (df['Clock'] <= e_date)
        newdf = df.loc[newdf]
        newdf['Clock'] = df['Clock'] + pd.DateOffset(years=1)
        return newdf

    except Exception as e:
        print(e)


@ensure_annotations
def data_from_weather_api_old():
    ''' weather data'''
    logger.info("weather data fetching")
    try:
        l1 = []
        value = []
        start_date = "2022-11-18"
        end_date = "2023-11-18"
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude=28.58&longitude=77.33&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m,precipitation&timezone=auto"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        l1.append(data)

        data1 = l1[0]
        data_dict = data1['hourly']

        for i in range(len(data_dict['time'])):
            value.append({
                "Clock": data_dict['time'][i],
                "temp": data_dict['temperature_2m'][i],
                "humidity": data_dict['relative_humidity_2m'][i],
                "rain": data_dict['rain'][i],
                "cloud_cover": data_dict['cloud_cover'][i],
                "wind_speed": data_dict['wind_speed_10m'][i],

            })
        df = pd.DataFrame(value)
        df['Clock'] = pd.to_datetime(df['Clock'])
        df.set_index("Clock", inplace=True, drop=True)
        df["temp_diff"] = df['temp'] - df['temp'].shift(1)
        df.drop(['temp'], axis=1, inplace=True)
        df.fillna(value=0, inplace=True)
        # df.dropna(inplace=True)
        return df

    except Exception as e:
        print(e)


@ensure_annotations
def scaling_data(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled


@ensure_annotations
def data_from_weather_apiV1(startDate, endDate):
    ''' Fetch weather data from CSV file based on date range'''
    logger.info("Weather data fetching")
    try:
        df = pd.read_csv("weather.csv")
        df['Clock'] = pd.to_datetime(df['Clock'])

        # Filter data based on date range
        filtered_df = df[(df['Clock'] >= startDate) & (df['Clock'] <= endDate)]

        return filtered_df

    except Exception as e:
        print(e)


@ensure_annotations
def uom():
    try:
        sensor_ids = [
            '5f718b613291c7.03696209', '5f718c439c7a78.65267835',
            '614366bce31a86.78825897', '6148740eea9db0.29702291', '625fb44c5fb514.98107900',
            '625fb9e020ff31.33961816', '6260fd4351f892.69790282', '627cd4815f2381.31981050',
            '629094ee5fdff4.43505210', '62aad7f5c65185.80723547', '62b15dfee341d1.73837476',
            '62b595eabd9df4.71374208', '6349368c306542.16235883', '634e7c43038801.39310596',
            '6399a18b1488b8.07706749', '63a4195534d625.00718490', '63a4272631f153.67811394',
            '63aa9161b9e7e1.16208626', '63ca403ccd66f3.47133508', '62a9920f75c931.62399458'
        ]

        url = "https://multipoint.myxenius.com/Sensor_newHelper/getDataApi"
        params = {

            'sql': "SELECT id AS uuid, name AS sensorName, CASE WHEN grid_billing_type IS NOT NULL THEN grid_billing_type ELSE 'UOM' END AS uom FROM sensor WHERE id IN ({}) ORDER BY name".format(
                ','.join(f"'{sid}'" for sid in sensor_ids)),
            'type': 'query'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        sensor_list = [{'uuid': item['uuid'], 'sensorName': item['sensorName'], "UOM": item['uom']} for item in
                       data['resource']]
        df = pd.DataFrame(sensor_list)
        logger.info("UOM values")
        return df

    except Exception as e:
        print(e)


def sensor_data(id_lst):
    try:
        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection2")
        MONGO_URL = f"mongodb://{host}:{port}"

        '''Writing logs'''
        logger.info("Reading data from Mongo DB")

        '''Exception Handling'''
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]
        data_list = []
        for id in id_lst:
            value = list(collection.find(
                {"id": id},
                {"meter_ct_mf": 1, "UOM": 1, "meter_MWh_mf": 1, "site_id": 1, "asset_id": 1,
                 "sensor_id": "$parent_sensor_id"}
            ))
            data_list.extend(value)  # Use extend instead of append to flatten the list
        df = pd.DataFrame(data_list)
        return df
    except Exception as e:
        print(e)


@ensure_annotations
def add_lagsV1(df: pd.DataFrame) -> pd.DataFrame:
    try:
        target_map = df['consumed_unit'].to_dict()

        # 15 minutes, 30 minutes, 1 day, 7 days, 15 days
        df['lag1'] = df['consumed_unit'].shift(1)
        df['lag2'] = df['consumed_unit'].shift(2)
        df['lag3'] = df['consumed_unit'].shift(96)  # 96 periods in 1 day (24 hours * 4 quarters)
        df['lag4'] = df['consumed_unit'].shift(672)  # 672 periods in 7 days (7 days * 24 hours * 4 quarters)
        df['lag5'] = df['consumed_unit'].shift(1440)  # 1440 periods in 15 days (15 days * 24 hours * 4 quarters)

        # Fill missing values with zeros or NaNs
        df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5']] = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5']].fillna(0)
    except KeyError as e:
        print(f"Error: {e}. 'consumed_unit' column not found in the DataFrame.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    return df
