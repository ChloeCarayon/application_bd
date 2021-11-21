import click

from models.train_model import train_model_h2o, train_model_xboost
from features.build_features import generate_features
from models.predict_model import predict_xgboost, predict_h2o
from utils import directory_path
from data.make_dataset import generate_raw
from visualization.do_explicability_h2o import shap_visualisation_h2o
from visualization.do_explicability_xgboost import shap_visualisation_xgboost

@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",

    type=click.Choice(["generate_raw", "features", "generate_do_features", "train", "predict", "visualisation"]),
)
def do_classic_action(task):
    if task == "generate_raw":
        generate_raw("application_train.csv")
    if task == "features":
        generate_features(directory_path, 'TARGET')
    if task == "generate_do_features":
        generate_raw("application_train.csv")
        generate_features(directory_path, 'TARGET')
    if task == "train":
        get_parameters_training()
    if task == "predict":
        get_parameters_predict()
    if task == "visualisation":
        get_parameters_visualisation()


@click.command()
@click.option(
    "--model_type",
    prompt="Choose a model to train",
    type=click.Choice(["xgboost", "gmb", "rf", "automl", "XgboostClassifier"]),
)
@click.option("--version", prompt="Version number", default="0.0.1", type=str)
def get_parameters_training(model_type, version):
    if model_type =='XgboostClassifier':
        train_model_xboost(version)
    else:
        train_model_h2o(model_type, version)


@click.command()
@click.option(
    "--model_type",
    prompt="Choose a model to train",
    type=click.Choice(["xgboost", "gmb", "rf", "automl", "xgboostClassifier"]),
)
@click.option("--version", prompt="Version number", default="0.0.1", type=str)
def get_parameters_predict(model_type, version):
    if model_type == 'xgboostClassifier':
        predict_xgboost(version)
    else:
        predict_h2o(model_type, version)


@click.command()
@click.option(
    "--model_type",
    prompt="Choose a xgboost model for explicability",
    type=click.Choice(["xgboostH2o", "xgboostClassifier"]),
)
@click.option("--version", prompt="Version number", default="0.0.1", type=str)
@click.option("--point", prompt="Explanation on specific point", default=4, type=int)
def get_parameters_visualisation(model_type, version, point):
    if model_type == 'xgboostH2o':
        shap_visualisation_h2o(version, point)
    else:
        shap_visualisation_xgboost(version, point)

if __name__ == "__main__":
    do_classic_action()
