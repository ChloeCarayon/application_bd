import click

from models.train_model import train_model


@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",
    type=click.Choice(["preprocess", "features", "train", "predict"]),
)
def do_action(task):
    if task == "preprocess":
        print("preprocessing")
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
    do_action()
