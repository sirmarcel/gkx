"""Run NVT MD w/ ASE."""

from jax import lax, jit

import numpy as np
from ase.units import fs
from ase.md import Langevin
from pathlib import Path

from glp.ase import Calculator
from glp.instantiate import get_calculator
from glp.dynamics import atoms_to_input
from glp.dynamics.utils import Point

from gkx import comms
from gkx.utils.trees import tree_slice

from .dataset import chunk_to_dataset
from .batcher import to_batch, fake_batch
from .chunker import Chunker


def run(
    maxsteps,
    potential,
    calculator,
    dt,
    temperature,
    friction,
    initial_atoms,
    supercell_atoms=None,
    primitive_atoms=None,
    outfolder=Path("trajectory/"),
    chunk_size=2500,
    initial_step=0,
    initial_chunk=0,
):
    # todo: be more forgiving
    assert maxsteps % chunk_size == 0
    assert maxsteps > 1  # otherwise we won't save anything

    reporter = comms.reporter()
    reporter.start("running MD")
    reporter.step("setup")

    calculator = Calculator(get_calculator(potential, calculator), raw=True)

    atoms = initial_atoms.copy()
    atoms.calc = calculator

    dynamics = Langevin(atoms, dt * fs, temperature_K=temperature, friction=friction)

    chunker = Chunker(chunk_size)

    reporter.step("initial step")
    atoms.get_forces()

    reporter.step(f"md ", spin=False)
    n_step = initial_step
    n_chunk = initial_chunk
    state_str = "starting"

    if n_step == 0:
        # make sure to save the initial step
        chunker.submit(atoms_to_batch(atoms))

    while n_step < maxsteps:
        reporter.tick("🚀 " + state_str)

        dynamics.run(steps=1)
        chunk = chunker.submit(atoms_to_batch(atoms))

        time_per_step = reporter.timer_step() / chunker.count
        current_steps = chunker.count + initial_step
        remaining_steps = maxsteps - current_steps
        remaining_time = time_per_step * remaining_steps

        state_str = f"{current_steps}/{maxsteps} (ETA: {(remaining_time/60):.1f}min) ({time_per_step*1000:.0f}ms/step)"

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

            n_step += chunk_size
            n_chunk += 1

    reporter.tick("(✨) " + state_str)

    reporter.done()


def atoms_to_batch(atoms):
    point = Point(atoms.get_positions(), atoms.get_momenta())
    results = atoms.calc.results
    return fake_batch(point, results)
