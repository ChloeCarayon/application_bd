import click

from models.train_model import train_model


from features.build_features import generate_features
from utils import directory_path
from data.make_dataset import generate_raw

@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",

    type=click.Choice(["generate_raw", "features", "generate_do_features", "train", "predict"]),
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


@click.command()
@click.option(
    "--model_type",
    prompt="Choose a model to train",
    type=click.Choice(["xgboost", "gmb", "rf", "automl"]),
)
@click.option("--version", prompt="Version number", default="0.0.1", type=str)
def get_parameters_training(model_type, version):
    train_model(model_type, version)

if __name__ == "__main__":
    do_classic_action()
