# run a GK NVE

import numpy as np
import xarray as xr
from pathlib import Path
from copy import deepcopy

from ase.units import fs
from vibes import keys, dimensions

from gkx import comms
from gkx.utils import open_trajectory

from .logger import Logger

defaults = {
    "files": {
        "geometry": "geometry.in",
        "supercell": "geometry.in.supercell",
        "primitive": "geometry.in.primitive",
    },
    "batch_size": 25,
    "chunk_size": 2500,
    "workdir": Path("md/"),
    "maxsteps": 25000,
    "logger": {
        "every": 200,
        "outfile": "md.log",
        "quantities": [
            "steps",
            "time_ps",
            "ms_per_step",
            "temperature",
            "energy_total",
        ],
    },
    "x64": False,
}

defaults_recompute = deepcopy(defaults)
defaults_recompute["logger"] = {
    "every": 20,
    "outfile": "recompute.log",
    "quantities": [
        "steps",
        "ms_per_step",
    ],
}

defaults_nve_re_hf = deepcopy(defaults)
defaults_nve_re_hf["logger"] = {
    "every": 100,
    "outfile": "md.log",
    "quantities": [
        "steps",
        "time_ps",
        "ms_per_step",
        "temperature",
        "energy_total",
        "mean_squared_displacement",
    ],
}


def run_from_yaml(file):
    from specable.inout import read_yaml

    run_from_dict(read_yaml(file))


def run_from_dict(dct):
    from ase import io
    from ase import units

    if "recompute" not in dct and "nve_re_hf" not in dct:
        dct = {**defaults, **dct}
    elif "recompute" in dct:
        dct = {**defaults_recompute, **dct}
    elif "nve_re_hf" in dct:
        dct = {**defaults_nve_re_hf, **dct}

    x64 = dct.pop("x64")
    if x64:
        from jax import config

        config.update("jax_enable_x64", True)

        import jax.numpy as jnp

        dtype = jnp.float64
    else:
        import jax.numpy as jnp

        dtype = jnp.float32

    potential = dct.pop("potential")
    calculator = dct.pop("calculator")

    # todo: make exception for recompute?
    maxsteps = dct.pop("maxsteps")
    batch_size = dct.pop("batch_size")
    chunk_size = dct.pop("chunk_size")

    if maxsteps % chunk_size != 0:
        maxsteps = (maxsteps // chunk_size + 1) * chunk_size
        comms.warn(
            f"rounding up maxsteps to {maxsteps} (must be divisible by chunk_size)"
        )

    files = dct.pop("files")
    initial_atoms = io.read(files.pop("geometry"), format="aims")

    if "supercell" in files:
        supercell_atoms = io.read(files.pop("supercell"), format="aims")
    else:
        supercell_atoms = None

    if "primitive" in files:
        primitive_atoms = io.read(files.pop("primitive"), format="aims")
    else:
        primitive_atoms = None

    workdir = Path(dct.pop("workdir"))
    workdir.mkdir(exist_ok=True)

    logger_dct = dct.pop("logger")
    logger_dct["outfile"] = workdir / logger_dct["outfile"]

    logger = Logger(**logger_dct)

    if "recompute" in dct:
        name = dct.pop("outfolder", "recompute")
        outfolder = workdir / name
    else:
        name = dct.pop("outfolder", "trajectory")
        outfolder = workdir / name

    if outfolder.is_dir():
        if "recompute" not in dct:
            if (outfolder / "TMP").is_file():
                comms.warn("found TMP file; deleting. (did a job shut down hard?)")
                (outfolder / "TMP").unlink()

            n_chunks = len(list(outfolder.glob("*.nc")))
            if n_chunks > 0:
                traj = open_trajectory(outfolder)
                initial_step = len(traj.time)
                if initial_step >= maxsteps:
                    comms.talk("already done, quitting")
                    return None

                initial_atoms = traj.get_atoms(-1)
                initial_chunk = n_chunks
                comms.talk(
                    f"restarting from step {initial_step}, chunk {initial_chunk}"
                )
            else:
                initial_step = 0
                initial_chunk = 0

    else:
        outfolder.mkdir()
        initial_step = 0
        initial_chunk = 0

    if "nve" in dct:
        from .nve import run

        dt = dct["nve"]["dt"]

        run(
            maxsteps,
            potential,
            calculator,
            dt,
            initial_atoms,
            supercell_atoms=supercell_atoms,
            primitive_atoms=primitive_atoms,
            outfolder=outfolder,
            batch_size=batch_size,
            chunk_size=chunk_size,
            initial_step=initial_step,
            initial_chunk=initial_chunk,
            # workdir=workdir,
            logger=logger,
            dtype=dtype,
        )

    elif "nvt" in dct or "npt" in dct:
        from .nvt_npt import run

        run(
            maxsteps,
            potential,
            calculator,
            dct,
            initial_atoms,
            supercell_atoms=supercell_atoms,
            primitive_atoms=primitive_atoms,
            outfolder=outfolder,
            chunk_size=chunk_size,
            initial_step=initial_step,
            initial_chunk=initial_chunk,
            # workdir=workdir,
            logger=logger,
            dtype=dtype,
        )

    elif "nve_re_hf" in dct:
        from .nve_re_hf import run

        dt = dct["nve_re_hf"]["dt"]

        re_hf = dct.pop("nve_re_hf")
        full_frame_every = re_hf.pop("full_frame_every")
        calculator_heat_flux = re_hf.pop("calculator")
        batch_size_heat_flux = re_hf.pop("batch_size")
        heat_flux_every = re_hf.pop("every")

        run(
            maxsteps,
            potential,
            calculator,
            calculator_heat_flux,
            dt,
            initial_atoms,
            heat_flux_every=heat_flux_every,
            full_frame_every=full_frame_every,
            supercell_atoms=supercell_atoms,
            primitive_atoms=primitive_atoms,
            outfolder=outfolder,
            batch_size=batch_size,
            batch_size_heat_flux=batch_size_heat_flux,
            chunk_size=chunk_size,
            initial_step=initial_step,
            initial_chunk=initial_chunk,
            logger=logger,
            dtype=dtype,
        )

    elif "recompute" in dct:
        from .recompute import run

        assert str(dct["recompute"]["trajectory"]) != str(outfolder)

        trajectory = open_trajectory(workdir / dct["recompute"]["trajectory"])
        step_size = dct["recompute"]["step_size"]
        devices = dct["recompute"].get("devices", 1)

        if outfolder.is_dir():
            if (outfolder / "TMP").is_file():
                comms.warn("found TMP file; deleting. (did a job shut down hard?)")
                (outfolder / "TMP").unlink()

            n_chunks = len(list(outfolder.glob("*.nc")))
            if n_chunks > 0:
                traj = open_trajectory(outfolder)
                initial_step = len(traj)

                if (initial_step * step_size) >= len(trajectory.time):
                    comms.talk("already done, quitting")
                    return None

                initial_chunk = n_chunks
                initial_atoms = trajectory.get_atoms(initial_step * step_size)
                comms.talk(
                    f"restarting from (recompute) step {initial_step}, chunk {initial_chunk}"
                )
            else:
                initial_step = 0
                initial_chunk = 0

        run(
            trajectory,
            potential,
            calculator,
            step_size,
            initial_atoms,
            supercell_atoms=supercell_atoms,
            primitive_atoms=primitive_atoms,
            outfolder=outfolder,
            chunk_size=chunk_size,
            batch_size=batch_size,
            initial_step=initial_step,
            initial_chunk=initial_chunk,
            devices=devices,
            dtype=dtype,
            logger=logger,
        )
