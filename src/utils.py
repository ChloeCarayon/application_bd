import pickle
import os


directory_path = os.path.realpath(__file__).replace("src/utils.py", "")


def dump_pickle(path: str, object):
    with open(f"{path}.pickle", 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(f"{path}.pickle", 'rb') as handle:
        object = pickle.load(handle)
    return object




