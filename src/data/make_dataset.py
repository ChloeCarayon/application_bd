# -*- coding: utf-8 -*-

from kaggle.api.kaggle_api_extended import KaggleApi

from utils import directory_path, unzip_file


def authenticate_kaggle_api():
    """This function is used for
    Kaggle API authentication.
    """
    api = KaggleApi()
    api.authenticate()
    return api


def get_raw_kaggle(competition_name: str, file: str, path_file: str):
    """This function get the competition
    datasets from Kaggle API.

    :param competition_name: name of the competition
    :type competition_name: str
    :param file: name of the dataset
    :type file: str
    :param path_file: path defined to store the .zip
    :type path_file: str
    """
    api = authenticate_kaggle_api()
    api.competition_download_file(competition_name, file, path=path_file)


def generate_raw(file: str):
    """This function generates the csv dataset.

    :param file: name of the dataset
    :type file: str
    """
    competition_name = "home-credit-default-risk"
    path_file_zip = f"{directory_path}data/external/"
    path_file_csv = f"{directory_path}data/raw/"
    get_raw_kaggle(competition_name, file, path_file_zip)
    unzip_file(f"{path_file_zip}{file}", f"{path_file_csv}")
    print('Done')

