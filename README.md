# gkx: Green-Kubo Method in JAX

The Green-Kubo method is an approach to compute the thermal conductivity of materials, in particular those with highly anharmonic potential-energy surfaces, through equilibrium molecular dynamics simulations. In two [recent](https://marcel.science/gknet/) [preprints](https://marcel.science/glp/), we introduced an approach to use "modern" machine-learning potentials which use message passing and automatic differentiation for this method.

This repository provides a barebones implementation of the GK method with `jax`, based on `vibes` (for the actual GK functionality, i.e., computing thermal conductivities), `stepson` (post-processing infrastructure), `glp` (heat flux and MD), and `mlff` (for the so3krates potential). The high-cost parts of the method, long-running NVE molecular dynamics, are executed entirely on the GPU, only occasionally returning to CPU to write results. For convenience, we also have a wrapper to the `ase` Langevin thermostat for thermalising systems.

**This is early-stage code, factored ouf of the [`glp-archive`](https://github.com/sirmarcel/glp-archive) paper repository. It works, but it has not yet been cleaned up and optimised for public usage. Performance is okay, but we leave a bunch of optimisations on the table for now. Please regard as a demo, not a finished product!**

Nevertheless, *look at it go*:

[![asciicast](https://asciinema.org/a/R1SafF6GYW9tNGdUhPRVLNkAE.svg)](https://asciinema.org/a/R1SafF6GYW9tNGdUhPRVLNkAE?t=0:22)

This runs 0.1ns of MD with 512 atoms with the LJ potential. With a so3krates potential, speed is reduced by about an order of magnitude, so let's not get too excited!

## Installation

The most difficult part will be getting the dependencies to run: You need a CUDA-ready install of `jax`, and then `glp` and `mlff`. Once that's done, `pip install .` in a clone of the repository should suffice, which will install the remaining dependencies.

## Interface

`gkx` is currently extremely minimalistic. It has exactly two CLI commands:

- `gkx run md input.yaml` loads instructions from `input.yaml` and then runs MD
- `gkx out gk trajectory/` performs the Green-Kubo processing and emits a `greenkubo.nc` dataset with the result

For additional information, until docs are written, please refer to the CLI itself or, even better, the code, for further details.

Input files look like this:

```yaml
nve:
  # timestep in fs
  dt: 4.0
# number of timesteps
maxsteps: 250000
# size of written output chunks (must divide maxsteps)
chunk_size: 25000
# number of steps performed on the GPU in one call
# may need to be adapted to match available RAM
batch_size: 25

files:
  # starting configuration (FHI-aims input format)
  geometry: geometry.in
  # "pristine" supercell before thermalisation (optional)
  supercell: geometry.in.supercell
  # primitive cell (optional)
  primitive: geometry.in.primitive


# matching to the glp.instantiate conventions:
potential:
  lennard_jones:
    sigma: 3.405
    epsilon: 0.01042
    cutoff: 9.0
    onset: 8.0

calculator:
  atom_pair:
    heat_flux: True

```

To run thermalisation, the `nve` block is replaced with

```yaml
nvt:
  temperature: 10
  dt: 4.0
  friction: 0.02
```

This simply uses the `ase` implementation of the Langevin thermostat.

## Units

We defer to `ase` and `vibes` for units. Internally, everything is based on `ase`, so the timestep is internally in `ase.units.fs` rather than SI `fs`. The datasets that are written respect the `FHI-vibes` conventions, which use `ase` for everything except the heat flux, which is written in SI units and uses `ps` (rather than `fs`) as the base time unit. Stress and heat flux are divided by the system volume.
