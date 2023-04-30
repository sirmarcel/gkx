from gkx.utils.trees import tree_slice, tree_concatenate

from .utils import get_length

class Chunker:
    """Chunker utility class.

    During MD, we get batches of potentially differing sizes
    due to overflows. This class keeps track of a queue of such
    batches, and whenever we reach the `chunk_size`, we emit a
    chunk of that exact size.

    For convenience, we also count the total number of steps.

    """

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.count = 0

        self.hold = []
        self.hold_length = 0

    def submit(self, batch):
        length = get_length(batch)
        self.count += length

        if self.hold_length + length >= self.chunk_size:
            taken = self.chunk_size - self.hold_length

            main = tree_slice(batch, slice(0, taken))
            rest = tree_slice(batch, slice(taken, None))

            chunk = tree_concatenate(self.hold + [main])

            self.hold = [rest]
            self.hold_length = length - taken

            return chunk

        else:
            self.hold.append(batch)
            self.hold_length += length

            return None
