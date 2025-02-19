from collections import namedtuple
from pathlib import Path

Quantity = namedtuple("Quantity", ("name", "extract", "format", "width"))


def get_temperature(state):
    if state.last_dataset:
        return state.last_dataset.temperature.mean().compute()
    else:
        return None


def format_temperature(T):
    if T:
        return f"{T:.1f}"
    else:
        return "--"


def format_energy_total(etot):
    if etot:
        return f"{etot:12.6f}"
    else:
        return "--"


def format_msd(msd):
    if msd:
        return f"{msd:4.4f}"
    else:
        return "--"


def get_energy_total(state):
    if state.last_dataset:
        e_kin = state.last_dataset.energy_kinetic.mean().compute()
        e_pot = state.last_dataset.energy_potential.mean().compute()
        return e_kin + e_pot
    else:
        return None


def get_msd(state):
    if state.last_dataset:
        if "mean_squared_displacement" in state.last_dataset:
            return state.last_dataset.mean_squared_displacement.mean().compute()
        else:
            return None
    else:
        return None


all_quantities = {
    "steps": Quantity(
        "steps",
        lambda state: state.current_steps,
        lambda x: f"{x}",
        10,
    ),
    "time_ps": Quantity(
        "time/ps",
        lambda state: state.simulation_time / 1000,
        lambda x: f"{x:.1f}",
        8,
    ),
    "ms_per_step": Quantity(
        "ms/step", lambda state: state.time_per_step, lambda x: f"{1000*x:.1f}", 8
    ),
    "temperature": Quantity("T/K", get_temperature, format_temperature, 8),
    "energy_total": Quantity("Etot/eV", get_energy_total, format_energy_total, 12),
    "mean_squared_displacement": Quantity("MSD/Å²", get_msd, format_msd, 8),
}


class Logger:
    def __init__(
        self,
        every=100,
        quantities=["time_ps", "ms_per_step"],
        outfile=Path("md.log"),
    ):
        self.every = every
        self.quantities = [all_quantities[q] for q in quantities]
        self.outfile = outfile

        self.last = None

    def __call__(self, state):
        if self.last is None:
            if not self.outfile.is_file():
                self.outfile.touch()
                self.write_header()

            self.last = state.current_steps
            self.log(state)

        if self.last + self.every <= state.current_steps:
            self.log(state)
            self.last = state.current_steps

    def log(self, state):
        row = [f"{q.format(q.extract(state)):>{q.width}}" for q in self.quantities]
        self.write_row(row)

    def write_row(self, row):
        with open(self.outfile, "a") as f:
            f.write(" ".join(row) + "\n")

    def write_header(self):
        with open(self.outfile, "a") as f:
            f.write(
                "Note: Derived quantities (T,E,...) are averaged over the last chunk/batch. Exact information can be found in the trajectory!"
                + "\n\n\n"
            )

        headers = [f"{quantity.name:>{quantity.width}}" for quantity in self.quantities]

        self.write_row(headers)


class ReportingState:
    def __init__(self, dt, initial_step, initial_chunk, maxsteps, chunker, step_size=1):
        self.dt = dt
        self.initial_step = initial_step
        self.initial_chunk = initial_chunk
        self.maxsteps = maxsteps
        self.chunker = chunker
        self.step_size = step_size

        self.time_per_step = None
        self.current_steps = None
        self.remaining_steps = None
        self.remaining_time = None
        self.simulation_time = None
        self.last_dataset = None

    def update(self, time, last_dataset=None):
        self.time_per_step = time / (self.chunker.count * self.step_size)
        self.current_steps = (self.chunker.count + self.initial_step) * self.step_size
        self.remaining_steps = self.maxsteps - self.current_steps
        self.remaining_time = self.time_per_step * self.remaining_steps
        self.simulation_time = self.current_steps * self.dt

        if last_dataset is not None:
            self.last_dataset = last_dataset

    @property
    def state_str(self):
        if self.current_steps is not None:
            return f"{self.current_steps}/{self.maxsteps} (ETA: {(self.remaining_time/60):.1f}min) ({self.time_per_step*1000:.0f}ms/step)"
        else:
            return "starting"
