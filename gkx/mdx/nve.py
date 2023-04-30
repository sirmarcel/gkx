"""Run NVE MD."""

from jax import lax, jit

import numpy as np
from ase.units import fs
from pathlib import Path

from glp.instantiate import get_dynamics
from glp.dynamics import atoms_to_input

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
    initial_atoms,
    supercell_atoms=None,
    primitive_atoms=None,
    outfolder=Path("trajectory/"),
    batch_size=25,
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

    dynamics_fn = get_dynamics(potential, calculator, {"verlet": {"dt": dt * fs}})
    initial = atoms_to_input(initial_atoms)
    dynamics, state = dynamics_fn(*initial)
    step_fn = jit(dynamics.step)

    chunker = Chunker(chunk_size)

    reporter.step(f"md ", spin=False)
    n_step = initial_step
    n_chunk = initial_chunk
    state_str = "starting"

    if n_step == 0:
        # make sure to save the initial step
        chunker.submit(fake_batch(state.point, state.results))

    while n_step < maxsteps:
        reporter.tick("🚀 " + state_str)
        new_state, outputs = lax.scan(step_fn, state, None, length=batch_size)

        if new_state.overflow:
            good_until = np.argmax(outputs[2] == True)  # find first overflow
            if good_until == 0:
                comms.warn("overflow occured immediately")
                comms.warn(
                    "if this occurs repeatedly, skin+cutoff may be too small for cell"
                )
                # can't salvage this batch
                initial = state.point
                dynamics, state = dynamics.update(initial)
                step_fn = jit(dynamics.step)
                batch = None
            else:
                # salvage non-overflow parts of batch
                initial = tree_slice(outputs[0], good_until - 1)
                dynamics, state = dynamics.update(initial)
                step_fn = jit(dynamics.step)

                batch = to_batch(outputs, length=good_until)

        else:
            batch = to_batch(outputs)
            state = new_state

        if batch is not None:
            chunk = chunker.submit(batch)

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
