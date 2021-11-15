import click

from data.make_dataset import generate_raw


@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",
    type=click.Choice(["generate_raw", "preprocess", "features", "train", "predict"]),
)
def do_action(task):
    if task == "generate_raw":
        generate_raw("application_train.csv")


if __name__ == "__main__":
    do_action()
