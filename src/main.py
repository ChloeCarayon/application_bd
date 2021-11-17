import click

from features.build_features import build_features, generate_features
from utils import directory_path


@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",
    type=click.Choice(["create_dataset", "preprocess", "features", "train", "predict"]),
)
def do_action(task):
    if task == "features":
        generate_features(directory_path, 'TARGET')


if __name__ == "__main__":
    do_action()
