"""Tools to work with batches.

An `output` is what we get out of `scan`. We turn
it into a batch, which is just a dict of ndarrays.
"""

import numpy as np

from gkx.utils.trees import tree_slice, tree_unsqueeze


def fake_batch(point, results):
    points = tree_unsqueeze(point)
    results = tree_unsqueeze(results)

    return to_batch((points, results, None))


def to_batch(outputs, length=None):
    # scan output -> dict of ndarrays

    points, results, _ = outputs

    points = tree_slice(points, slice(0, length))
    results = tree_slice(results, slice(0, length))

    batch = {}

    # todo: make this more general purpose to also support cell, etc
    batch["positions"] = np.array(points.R)
    batch["momenta"] = np.array(points.P)

    for key, value in results.items():
        batch[key] = np.array(value)

    return batch
