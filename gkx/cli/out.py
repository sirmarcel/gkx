import click
from pathlib import Path
import xarray as xr
import numpy as np

# if this is smaller than the number of files,
# we get extremely verbose (but ultimately harmless)
# warnings, see https://github.com/pydata/xarray/issues/7549
xr.set_options(file_cache_maxsize=512)

from vibes import defaults, keys

from stepson.trajectory.derived_properties import getter, add_property


@click.group()
def out():
    """out"""


@out.command()
@click.argument("file", default=Path("trajectory/"), type=Path)
@click.option("-o", "--outfile", default="greenkubo.nc", type=Path)
@click.option("--outfolder", default=Path("."), type=Path)
@click.option(
    "--maxsteps", default=None, type=int, help="cut off dataset after maxsteps"
)
@click.option("--offset", default=None, type=int, help="start from offset")
@click.option("--spacing", default=None, type=int, help="use only every nth step")
@click.option("--freq", default=1.0, type=float, help="lowest characteristic frequency")
def gk(file, outfile, outfolder, maxsteps, offset, spacing, freq):
    """perform greenkubo analysis for heat flux dataset in FILE"""
    from stepson import comms
    from stepson.green_kubo import get_kappa_dataset
    from stepson.utils import open_dataset

    reporter = comms.reporter()
    reporter.start(f"working on {file}")

    outfolder.mkdir(exist_ok=True)
    outfile = outfolder / outfile

    if offset is not None:
        if offset < 0:
            dataset = open_dataset(file)
            offset = len(dataset.time) + offset

    if offset is not None:
        outfile = outfile.parent / f"{outfile.stem}.from_{offset}.nc"

    if maxsteps is not None:
        outfile = outfile.parent / f"{outfile.stem}.to_{maxsteps}.nc"

    if spacing is not None:
        outfile = outfile.parent / f"{outfile.stem}.every_{spacing}.nc"

    if freq is not None:
        outfile = outfile.parent / f"{outfile.stem}.freq_{freq:.2f}.nc"

    if outfile.is_file():
        comms.warn(f"{outfile} exists, skipping")
        reporter.done()
        return None

    dataset = open_dataset(file)

    if maxsteps is not None:
        comms.talk(f"truncating to {maxsteps} timesteps")

        if len(dataset.time) < maxsteps:
            comms.warn(
                f"Tried to truncate {len(dataset.time)} timesteps to {maxsteps}, but dataset too short."
            )
            reporter.done()
            return None

        dataset = dataset.isel(time=slice(0, maxsteps))

    if offset is not None:
        comms.talk(f"starting from timestep {offset}")
        dataset = dataset.isel(time=slice(offset, len(dataset.time)))

    if spacing is not None:
        comms.talk(f"using spacing {spacing}")
        dataset = dataset.isel(time=slice(0, len(dataset.time), spacing))

    ds_gk = get_kappa_dataset(
        dataset,
        window_factor=1.0,
        aux=False,
        freq=freq,
    )

    reporter.step(f"write to {outfile}")

    ds_gk.to_netcdf(outfile)

    reporter.done()
