"""Run NVE MD that computes HF only for some steps."""

from jax import lax, jit
import jax.numpy as jnp

import numpy as np
from ase.units import fs
from pathlib import Path

from stepson.trajectory import Trajectory

from glp.instantiate import get_dynamics, get_calculator
from glp.dynamics import atoms_to_input

from gkx import comms
from gkx.utils.trees import tree_slice, tree_unsqueeze

from .dataset import chunk_to_dataset, data_to_array
from .batcher import to_batch, fake_batch
from .chunker import Chunker
from .logger import Logger, ReportingState
from .diagnostics import print_diagnostics


def run(
    maxsteps,
    potential,
    calculator,  # calculator for MD
    calculator_heat_flux,
    dt,
    initial_atoms,
    heat_flux_every=10,
    full_frame_every=500,
    supercell_atoms=None,
    primitive_atoms=None,
    outfolder=Path("trajectory/"),
    batch_size=100,  # batch_size MD
    batch_size_heat_flux=50,
    chunk_size=2500,  # chunk_size HF
    initial_step=0,
    initial_chunk=0,
    logger=Logger(),
    dtype=jnp.float32,
    keys_to_keep=[
        "temperature",
        "energy_total",
        "energy_potential",
        "energy_kinetic",
        "pressure",
        "pressure_kinetic",
        "pressure_potential",
        "heat_flux",
        "heat_flux_potential",
        "heat_flux_convective",
        "mean_squared_displacement",
    ],
    keys_to_keep_for_full_frame=[
        "positions",
        "velocities",
        "forces",
        "energy_potential",
    ],
):
    assert maxsteps % chunk_size == 0
    assert maxsteps > 1  # otherwise we won't save anything
    assert batch_size % heat_flux_every == 0  # don't want to batch unevenly

    assert full_frame_every % heat_flux_every == 0  # don't want to slice twice
    assert full_frame_every >= heat_flux_every
    full_frame_spacing = int(full_frame_every / heat_flux_every)

    assert chunk_size % full_frame_spacing == 0

    chunk_size_md = batch_size_heat_flux * heat_flux_every

    if initial_step * heat_flux_every > maxsteps:
        comms.talk("already done!")
        return None

    reporter = comms.reporter()
    reporter.start("running MD/HF")
    reporter.step("setup")

    dynamics_fn = get_dynamics(potential, calculator, {"verlet": {"dt": dt * fs}})
    initial = atoms_to_input(initial_atoms, dtype=dtype)

    dynamics, md_state = dynamics_fn(*initial)
    step_fn = jit(dynamics.step)  # otherwise every call to scan would re-trace
    chunker_md = Chunker(chunk_size_md)

    heat_flux, hf_state = get_hf_scan(
        get_calculator(potential, calculator_heat_flux),
        initial_atoms,
        batch_size_heat_flux,
        dtype=dtype,
    )
    chunker_hf = DatasetChunker(chunk_size)

    reporting_state = ReportingState(
        dt, initial_step, initial_chunk, maxsteps, chunker_hf, step_size=heat_flux_every
    )

    comms.talk("diagnostics for MD:")
    print_diagnostics(md_state=md_state)
    comms.talk("diagnostics for HF:")
    print_diagnostics(calc_state=hf_state, atoms=initial_atoms)

    n_step = initial_step  # the number of steps we have recomputed with the HF (not MD)
    n_chunk = initial_chunk
    dataset = None

    reporter.step("🚀", spin=False)
    if n_step == 0:
        # make sure to save the initial step
        chunker_md.submit(
            (tree_unsqueeze(md_state.point), tree_unsqueeze(md_state.results), None)
        )

    # complicated main loop: we run MD and then recompute certain frames with another calculator!
    # we try to keep the GPU fed, and so make use of jax asynchronous dispatch. comments starting
    # -- are designed to keep track of the work implicitly happening in the background

    while n_step * heat_flux_every <= maxsteps:
        reporter.tick(" " + reporting_state.state_str)
        new_md_state, outputs = lax.scan(step_fn, md_state, None, length=batch_size)
        # -- md is running now

        if dataset is not None:  # triggers if we have previously started computing HF
            # -- hf is still running, and the next batch of MD should be in the queue
            #    -> we can block until the HF is done, and still not run out of GPU work

            # -- hf is stopped here
            assert not new_hf_state.overflow
            hf_state = new_hf_state

            # write heat_flux into dataset
            stuff = to_batch(hf_outputs)
            for key, value in stuff.items():
                if "heat_flux" in key:
                    k, d = data_to_array(
                        key,
                        value,
                        {"time": dataset.time.data},
                        supercell_atoms,
                        len(value),
                    )
                    dataset[k] = d

            chunk_hf = chunker_hf.submit(dataset)
            reporting_state.update(reporter.timer_step(), last_dataset=dataset)
            logger(reporting_state)

            dataset = None

            # we could try to throw this into another round of the event loop, but KISS
            if chunk_hf is not None:
                # we drop most positions/velocities
                idx = np.arange(
                    1, len(chunk_hf.time) + 1
                )  # make sure that the last step is included (for restarts!)
                for key in keys_to_keep_for_full_frame:
                    chunk_hf[key].loc[idx % full_frame_spacing != 0] = float("nan")

                tmpfile = outfolder / "TMP"
                chunk_hf.to_netcdf(
                    tmpfile,
                    encoding={
                        k: dict(zlib=True, complevel=5)  # exploit the nans from above
                        for k in keys_to_keep_for_full_frame
                    },
                )
                tmpfile.replace(outfolder / f"{n_chunk:06d}.nc")

                comms.talk(
                    f"completed chunk {n_chunk} "
                    + f"(ETA: {(reporting_state.remaining_time/60):.1f}min) ({reporting_state.time_per_step*1000:.0f}ms/step)",
                    full=True,
                )

                n_chunk += 1

        # -- md is stopping
        assert not new_md_state.overflow
        md_state = new_md_state
        chunk_md = chunker_md.submit(outputs)

        # -- nothing is running right now, but this is quick

        if chunk_md is None:
            # -- either we re-enter here and submit a new MD job
            continue
        else:
            # -- or we submit hf work here
            thinned = tree_slice(
                chunk_md, slice(heat_flux_every - 1, None, heat_flux_every)
            )  # make sure the last step is included

            inputs_for_hf = thinned[0]
            new_hf_state, hf_outputs = heat_flux(hf_state, inputs_for_hf)

            # -- hf is now running
            #    -> we can use this time to build the dataset & compute diagnostics

            dataset = Dataset(
                chunk_to_dataset(
                    to_batch(thinned),
                    n_step,
                    dt * heat_flux_every,
                    initial_atoms,
                    supercell_atoms=supercell_atoms,
                    primitive_atoms=primitive_atoms,
                )
            )  # awkward, but we need the stepson tools to compute T, etc.

            n_step += len(dataset.time)

            dataset.temperature.compute()
            dataset.pressure.compute()
            dataset.data["mean_squared_displacement"] = (
                (dataset.displacements**2).mean(dim=("I", "a")).compute()
            )
            dataset = dataset.data  # back to bare xarray

            for key in dataset.keys():
                if key not in (keys_to_keep + keys_to_keep_for_full_frame):
                    dataset = dataset.drop(key)

    # we're done -- log and quit!
    reporting_state.update(reporter.timer_step())
    logger.log(reporting_state)

    reporter.tick("(✨) " + reporting_state.state_str)
    reporter.done(f"done at {reporting_state.time_per_step*1000:.0f}ms/step")


def get_hf_scan(calculator_fn, atoms, batch_size, dtype=jnp.float32):
    from glp.dynamics.utils import to_velocities, to_system

    system, velocities, masses = atoms_to_input(atoms, dtype=dtype)
    calculator, state = calculator_fn(system)

    to_V = lambda P: to_velocities(P, masses)
    to_S = lambda point: to_system(point, system)

    @jit
    def calculate(prior_state, point):
        results, new_state = calculator.calculate(
            to_S(point),
            prior_state,
            velocities=to_V(point.P),
            masses=masses,
        )

        return new_state, (point, results, new_state.overflow)

    def scanner(state, input_batch):
        return lax.scan(calculate, state, input_batch, length=batch_size)

    return scanner, state


class DatasetChunker:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.count = 0

        self.hold = []
        self.hold_length = 0

    def submit(self, dataset):
        import xarray as xr

        length = len(dataset.time)
        self.count += length

        if self.hold_length + length >= self.chunk_size:
            chunk = xr.concat(self.hold + [dataset], "time")

            self.hold = []
            self.hold_length = 0

            return chunk

        else:
            self.hold.append(dataset)
            self.hold_length += length

            return None


# -- horrible!
# this is a hack, we need a way to make a stepson dataset without writing to disk

from json import loads
from vibes.helpers.converters import dict2atoms
from vibes import keys


class Dataset(Trajectory):
    def __init__(self, dataset):
        self.data = dataset
        self.atoms_template = loads(self.data.attrs[keys.reference_atoms])
        self.reference_atoms = dict2atoms(self.atoms_template)
