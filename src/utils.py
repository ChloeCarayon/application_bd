import pickle
import os


directory_path = os.path.realpath(__file__).replace("src/utils.py", "")


def dump_pickle(path: str, object):
    """This function dump pickle

    :param path: path
    :type path: str
    """
    with open(f"{path}.pickle", 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    """This function load pickle

    :param path: path
    :type path: str
    :return:
    :rtype: object
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




