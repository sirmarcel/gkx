## `mdx`: a barebones MD package

This is a *very* rough draft of a `glp`-backed molecular dynamics code. Currently, we only support `nve`, and `nvt` via `ase`, and are particularly focused on GK-MD.

### Formats, etc

Input files are: `.yaml` for overall specification of the task. `.in` (i.e. `FHI-aims` format) for geometries, velocities, etc.

Outputs are written as subsequently numbered `.nc` files of uniform size to a specified output directory. They can be conveniently opened with `gkx.utils.open_trajectory`, which will yield a `stepson.Trajectory` object. The `data` member of that is just a view on the dataset.

### Terminology

- `batch`: bunch of timesteps that get run in one go on the GPU with `lax.scan`
- `chunk`: larger bunch of timesteps that get saved to disk as one dataset. We try very hard to make them uniform in size. Chunk size should be picked such that they fit easily into memory, and such that recomputing them is not ridiculously expensive, as this is also the level at which we implement restarts.
