'''
Run the modules. Module analysis doesn't work from here yet.
'''
import click
import logging
from Software.model.model import run as model_main
from Software.plotting.analysis import run as analysis_main


tasks = {
    "model": model_main,
    "analysis": analysis_main,
}


logger = logging.getLogger(__name__)


def main(task):
    try:
        tasks[task]()
    except:
        logger.error(f"Task {task} failed")
        raise


@click.command()
@click.option(
    "--task",
    type=click.Choice(tasks.keys()),
    required=True,
    help="Name of task to execute",
)
def main_cli(task):
    main(task)