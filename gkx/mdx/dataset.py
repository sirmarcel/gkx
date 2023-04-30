import numpy as np
import xarray as xr

from .utils import get_length


def chunk_to_dataset(
    chunk, start_idx, dt, initial_atoms, supercell_atoms=None, primitive_atoms=None
):
    from stepson.trajectory import add_property
    from stepson.green_kubo.heat_flux import get_prefactor

    from vibes.helpers.converters import atoms2json
    from vibes import keys

    if supercell_atoms is not None:
        reference_atoms = supercell_atoms
    else:
        reference_atoms = initial_atoms

    length = get_length(chunk)

    time = dt * np.arange(start_idx, start_idx + length, dtype=float)
    time = {keys.time: time}

    chunk["velocities"] = chunk["momenta"] / reference_atoms.get_masses()[None, :, None]
    del chunk["momenta"]

    final_data = {}
    for key, value in chunk.items():
        k, d = data_to_array(key, value, time, reference_atoms, length)
        final_data[k] = d

    data = xr.Dataset(final_data, coords=time)

    attrs = {}
    attrs[keys.reference_atoms] = atoms2json(reference_atoms, reduce=False)
    attrs["natoms"] = len(reference_atoms)
    attrs[keys.time_unit] = "fs"
    attrs["symbols"] = reference_atoms.get_chemical_symbols()
    attrs["masses"] = reference_atoms.get_masses()
    attrs["timestep"] = float(dt)

    if supercell_atoms:
        attrs[keys.reference_supercell] = atoms2json(supercell_atoms, reduce=False)

    if primitive_atoms:
        attrs[keys.reference_primitive] = atoms2json(primitive_atoms, reduce=False)

    attrs[keys.volume] = reference_atoms.get_volume()

    data.attrs = attrs

    add_property(data, keys.temperature)

    attrs["prefactor"] = get_prefactor(attrs[keys.volume], data.temperature.mean().data)

    data.attrs = attrs

    return data


def data_to_array(key, data, time, reference_atoms, length):
    from ase.units import fs

    dims = guess_dimensions(data, len(reference_atoms), length)
    key = guess_key(key)

    # apply vibes conventions
    if "stress" in key:
        data = data / reference_atoms.get_volume()

    if "heat_flux" in key:
        data = 1000 * fs * data / reference_atoms.get_volume()

    return key, xr.DataArray(data, dims=dims, coords=time, name=key)


def guess_dimensions(data, n_atoms, length):
    from vibes import dimensions

    shape = data.shape
    if shape == (length, 3):
        return dimensions.time_vec
    elif shape == (length,):
        return dimensions.time
    elif shape == (length, 3, 3):
        return dimensions.time_tensor
    elif shape == (length, n_atoms, 3):
        return dimensions.time_atom_vec
    else:
        raise ValueError(f"unknown shape {shape}")


def guess_key(key):
    from vibes import keys

    if key == "energy":
        return keys.energy_potential
    elif key == "stress":
        return keys.stress_potential
    else:
        return key
