"""
Run the modules.

This module defines:
- main(task, outputfile): internal function that dispatches to the tasks.
- main_cli(): command line interface using click.

Usage examples:

    fr-mcmc --task mcmc
    fr-mcmc --task analysis --outputfile results.npz
"""

from __future__ import annotations

import logging

import click

from fr_mcmc.mcmc.mcmc import run as mcmc_main
from fr_mcmc.plotting.analysis import run as analysis_main

logger = logging.getLogger(__name__)

# Mapeo de nombres de tarea a funciones
tasks = {
    "mcmc": mcmc_main,
    "analysis": analysis_main,
}


def main(task: str, outputfile: str | None = None) -> None:
    """
    Ejecuta la tarea seleccionada.

    Parameters
    ----------
    task : {"mcmc", "analysis"}
        Nombre de la tarea a ejecutar.
    outputfile : str or None
        Ruta del archivo a analizar (sólo relevante para "analysis").
    """
    if task == "mcmc":
        try:
            tasks[task]()
        except Exception:
            logger.error("Task %s failed", task)
            raise

    elif task == "analysis":
        try:
            tasks[task](outputfile)
        except Exception:
            logger.error("Task %s failed", task)
            raise

    else:
        raise ValueError(f"Unknown task: {task!r}")


@click.command()
@click.option(
    "--task",
    type=click.Choice(tasks.keys()),
    required=True,
    help="Name of task to execute",
)
@click.option(
    "--outputfile",
    type=str,
    required=False,
    help="File to analyze (only for task='analysis').",
)
def main_cli(task: str, outputfile: str | None) -> None:
    """
    Interfaz de línea de comandos.

    Esta función es el entry point tanto para:

        python -m fr_mcmc --task ...
        fr-mcmc --task ...

    gracias a la integración con click.
    """
    main(task, outputfile)
