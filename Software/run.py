'''
Run the modules. 
'''

import click
import logging
from Software.mcmc.mcmc import run as mcmc_main
from Software.plotting.analysis import run as analysis_main


tasks = {
    "mcmc": mcmc_main,
    "analysis": analysis_main,
}



logger = logging.getLogger(__name__)

def main(task, outputfile):
	if task == "mcmc":
		try:
			tasks[task]()
		except:
		    logger.error(f"Task {task} failed")
		    raise

	elif task == "analysis":
		try:
			tasks[task](outputfile)
		except:
		    logger.error(f"Task {task} failed")
		    raise


@click.command()
@click.option("--task", type=click.Choice(tasks.keys()), required=True, help="Name of task to execute")
@click.option("--outputfile", type=str, required=False, help="File to analize")



def main_cli(task, outputfile=''):
    main(task, outputfile)
