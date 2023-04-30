import click
from pathlib import Path
import xarray as xr
import numpy as np


@click.group()
def run():
    """run"""


@run.command()
@click.argument("infile", default=Path("md.yaml"), type=Path)
def md(infile):
    from gkx.mdx import run_from_yaml

    run_from_yaml(infile)
