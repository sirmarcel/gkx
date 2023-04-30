import click

@click.group()
def gkx():
    ...

from .out import out
gkx.add_command(out)

from .run import run
gkx.add_command(run)