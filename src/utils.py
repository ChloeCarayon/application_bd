import pickle
import os
import h2o
import pandas as pd
import json

directory_path = os.path.realpath(__file__).replace("src/utils.py", "")


def dump_json_object(path:str, object):
    """This function allows you to dump an object to json format and store it

    :param path: path to store
    :type path: str
    :param object: object to store
    :type object:
    """
    with open(path, "w") as outfile:
        json.dump(object, outfile)

def load_json_object(path:str):
    """This function allows you to load an object from json format

    :param path: path
    :type path: str
    :return: object
    :rtype:
    """
    with open(path) as json_file:
        object = json.load(json_file)
    return object

def dump_pickle(path: str, object):
    """This function allows you to dump an object to pickle format and store it

    :param path: path to store
    :type path: str
    :param object: object to store
    :type object:
    """
    with open(f"{path}.pickle", 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    """This function allows you to load an object from pickle format

    :param path: path to store
    :type path: str
    :return: object
    :rtype:
    """
    with open(f"{path}.pickle", 'rb') as handle:
        object = pickle.load(handle)
    return object

import zipfile

def unzip_file(old_path: str, new_path: str):
    """This function unzips a file.

    :param old_path: old path (for zip)
    :type old_path: str
    :param new_path: new path
    :type new_path: str
    """
    with zipfile.ZipFile(f"{old_path}.zip", "r") as zipref:
        zipref.extractall(new_path)


def get_model(model_type: str, version: str):
    """This function allows you to get the model

    :param model_type: model type
    :type model_type: str
    :param version: version of the model
    :type version: str
    :return: model
    :rtype:
    """
    path_model = f"{directory_path}models/{version}/{model_type}/{model_type}.zip"
    model = h2o.import_mojo(path_model)
    return model


def get_metadata():
    """This function allows you to get the metadata

    :return: metadata
    :rtype:
    """
    path_metadata = f"{directory_path}models/metadata"
    metadata = load_pickle(path_metadata)
    return metadata


def define_h2o_dataframe(df: pd.DataFrame, label: str):
    """This function transforms a pandas DataFrame
    into H2O DataFrame.
    It also defines the target column.

    :param df: dataset
    :type df: pd.DataFrame
    :param label: target
    :type label: str
    :return: a H2O DataFrame
    :rtype: h2o.H2OFrame
    """
    df_h2o = h2o.H2OFrame(df)
    df_h2o[label].asfactor().levels()
    df_h2o[label] = df_h2o[label].asfactor()
    return df_h2o
