import numpy as np
from unittest import TestCase

from gkx.utils.trees import *


class TestTrees(TestCase):
    def test_slice(self):
        for idx in [slice(0, 3), 1]:
            tree = {
                "test": np.array([1, 2, 3, 4]),
                "test2": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
            }

            sliced = tree_slice(tree, idx)

            np.testing.assert_array_equal(sliced["test"], tree["test"][idx])
            np.testing.assert_array_equal(sliced["test2"], tree["test2"][idx])

    def test_concatenate(self):
        tree = {
            "test": np.array([1, 2, 3, 4]),
            "test2": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        }
        tree2 = {"test": np.array([5, 6]), "test2": np.array([[3, 2], [3, 4]])}

        concatened = tree_concatenate([tree, tree2])

        np.testing.assert_array_equal(concatened["test"], np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(
            concatened["test2"],
            np.array([[1, 1], [2, 2], [3, 3], [4, 4], [3, 2], [3, 4]]),
        )

    def test_unsqueeze(self):
        tree = {"test": np.array([0])}
        unsqueezed = tree_unsqueeze(tree)

        np.testing.assert_array_equal(unsqueezed["test"], np.array([[0]]))
