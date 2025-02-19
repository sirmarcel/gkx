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

    def test_split_first_dim(self):
        tree = {
            "test": np.array([0, 1, 2, 3]),
            "test2": np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
        }

        target = {
            "test": np.array([[0, 1], [2, 3]]),
            "test2": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
        }

        split = tree_split_first_dim(tree, 2)

        np.testing.assert_array_equal(split["test"], target["test"])
        np.testing.assert_array_equal(split["test2"], target["test2"])

        merged = tree_merge_first_dim(split)

        np.testing.assert_array_equal(merged["test"], tree["test"])
        np.testing.assert_array_equal(merged["test2"], tree["test2"])

