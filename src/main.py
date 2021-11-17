import click


from features.build_features import generate_features
from utils import directory_path
from data.make_dataset import generate_raw



@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",

    type=click.Choice(["generate_raw", "features", "generate_do_features", "train", "predict"]),
)
def do_action(task):
    if task == "generate_raw":
        generate_raw("application_train.csv")
    if task == "features":
        generate_features(directory_path, 'TARGET')
    if task == "generate_do_features":
        generate_raw("application_train.csv")
        generate_features(directory_path, 'TARGET')



if __name__ == "__main__":
    do_action()
