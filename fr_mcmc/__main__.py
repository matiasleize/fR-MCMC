"""
Entry point to run the package with:

    python -m fr_mcmc --task mcmc

The real CLI is defined in fr_mcmc.run.main_cli (using click).
"""

from .run import main_cli


def main() -> None:
    """
    Small wrapper to be able to register it as an entry point if desired.

    In practice, main() simply delegates to main_cli(),
    which is decorated with click and reads arguments from sys.argv.
    """
    main_cli()


if __name__ == "__main__":
    main()
