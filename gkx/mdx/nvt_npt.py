"""Run NVT/NPT MD w/ ASE."""

from jax import lax, jit
import jax.numpy as jnp

import numpy as np
from ase.units import fs
from ase.constraints import voigt_6_to_full_3x3_stress

from pathlib import Path
from collections import namedtuple

from glp.ase import Calculator
from glp.instantiate import get_calculator
from glp.dynamics import atoms_to_input
from glp.dynamics.utils import Point

from gkx import comms
from gkx.utils.trees import tree_slice

from .dataset import chunk_to_dataset
from .batcher import to_batch, fake_batch
from .chunker import Chunker
from .logger import Logger, ReportingState
from .diagnostics import print_diagnostics

PointWithCell = namedtuple("PointWithCell", ("R", "P", "cell"))


def run(
    maxsteps,
    potential,
    calculator,
    dynamics_config,
    initial_atoms,
    supercell_atoms=None,
    primitive_atoms=None,
    outfolder=Path("trajectory/"),
    chunk_size=2500,
    initial_step=0,
    initial_chunk=0,
    logger=Logger(),
    dtype=jnp.float32,
):
    # todo: be more forgiving
    assert maxsteps % chunk_size == 0
    assert maxsteps > 1  # otherwise we won't save anything

    reporter = comms.reporter()
    reporter.start("running MD")
    reporter.step("setup")

    calculator = Calculator(get_calculator(potential, calculator), raw=True, dtype=dtype)

    atoms = initial_atoms.copy()
    atoms.calc = calculator

    nvt = dynamics_config.get("nvt", None)
    npt = dynamics_config.get("npt", None)

    if nvt:
        comms.talk("running nvt")
        from ase.md import Langevin

        dt = nvt["dt"]
        temperature = nvt["temperature"]
        friction = nvt["friction"]
        dynamics = Langevin(
            atoms, dt * fs, temperature_K=temperature, friction=friction
        )

        is_npt = False

    if npt:
        comms.talk("running npt")
        if npt["homogeneous"]:
            from ase.md.nptberendsen import NPTBerendsen as Berendsen
        else:
            from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen as Berendsen

        dt = npt["dt"]
        taut = npt["taut"] * fs
        taup = npt["taup"] * fs
        compressibility = npt["compressibility"]
        temperature = npt["temperature"]
        pressure = npt["pressure"]

        dynamics = Berendsen(
            atoms,
            dt * fs,
            temperature_K=temperature,
            compressibility=compressibility,
            taut=taut,
            taup=taup,
            pressure=pressure,
        )

        is_npt = True
        atoms.calc.raw = False

    chunker = Chunker(chunk_size)
    reporting_state = ReportingState(dt, initial_step, initial_chunk, maxsteps, chunker)

    reporter.step("initial step")
    atoms.get_forces()

    reporter.step(f"md ", spin=False)
    n_step = initial_step
    n_chunk = initial_chunk
    state_str = "starting"

    if n_step == 0:
        # make sure to save the initial step
        chunker.submit(atoms_to_batch(atoms, is_npt=is_npt))
        print_diagnostics(atoms=atoms)

    while n_step < maxsteps:
        reporter.tick("🚀 " + state_str)

        dynamics.run(steps=1)
        chunk = chunker.submit(atoms_to_batch(atoms, is_npt=is_npt))

        reporting_state.update(reporter.timer_step())
        logger(reporting_state)

        state_str = reporting_state.state_str

        if chunk is None:
            continue
        else:
            reporter.tick("🔧 " + state_str)

            dataset = chunk_to_dataset(
                chunk,
                n_step,
                dt,
                initial_atoms,
                supercell_atoms=supercell_atoms,
                primitive_atoms=primitive_atoms,
            )

            reporter.tick("💾 " + state_str)
            # attempt at atomic write -- does this work?
            tmpfile = outfolder / "TMP"
            dataset.to_netcdf(tmpfile)
            tmpfile.replace(outfolder / f"{n_chunk:06d}.nc")

            reporting_state.update(reporter.timer_step(), last_dataset=dataset)

            comms.talk(
                f"completed chunk {n_chunk} "
                + f"(ETA: {(reporting_state.remaining_time/60):.1f}min) ({reporting_state.time_per_step*1000:.0f}ms/step)",
                full=True,
            )
            reporting_state.update(reporter.timer_step(), last_dataset=dataset)
            print_diagnostics(atoms=atoms)

            n_step += chunk_size
            n_chunk += 1

    reporting_state.update(reporter.timer_step())
    logger.log(reporting_state)

    reporter.tick("(✨) " + state_str)
    reporter.done(f"done at {reporting_state.time_per_step*1000:.0f}ms/step")

    reporter.done()


def atoms_to_batch(atoms, is_npt=False):
    if not is_npt:
        point = Point(atoms.get_positions(), atoms.get_momenta())
    else:
        point = PointWithCell(
            atoms.get_positions(), atoms.get_momenta(), atoms.get_cell().array
        )

    results = atoms.calc.results.copy()
    if is_npt:
        results["stress"] = (
            voigt_6_to_full_3x3_stress(results["stress"]) * atoms.get_volume()
        )

    return fake_batch(point, results)
