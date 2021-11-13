import click


@click.command()
@click.option(
    "--task",
    prompt="Choose a task to execute",
    type=click.Choice(["preprocess", "features", "train", "predict"]),
)
def do_action(task):
    if task == "preprocess":
        print("preprocessing")


if __name__ == "__main__":
    do_action()
