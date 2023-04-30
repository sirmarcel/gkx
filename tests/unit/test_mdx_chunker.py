import numpy as np
from unittest import TestCase

from gkx.mdx.chunker import Chunker


class TestChunker(TestCase):
    def test(self):
        chunk1 = {"positions": np.array([0, 1, 2])}
        chunk2 = {"positions": np.array([3, 4])}
        chunk3 = {"positions": np.array([5, 6, 7])}
        chunk4 = {"positions": np.array([8])}

        chunker = Chunker(4)

        chunk = chunker.submit(chunk1)

        assert chunk is None
        assert chunker.count == 3

        chunk = chunker.submit(chunk2)

        assert chunker.count == 5
        np.testing.assert_array_equal(chunk["positions"], np.array([0, 1, 2, 3]))

        chunk = chunker.submit(chunk3)

        assert chunker.count == 8
        np.testing.assert_array_equal(chunk["positions"], np.array([4, 5, 6, 7]))

        chunk = chunker.submit(chunk4)

        assert chunk is None
