def get_length(chunk_or_batch_or_outputs):
    if isinstance(chunk_or_batch_or_outputs, tuple):
        if len(chunk_or_batch_or_outputs) == 3:
            points, _, _ = chunk_or_batch_or_outputs
            return len(points.R)
        else:
            # todo: make more friendly
            raise ValueError

    elif "positions" in chunk_or_batch_or_outputs:
        return len(chunk_or_batch_or_outputs["positions"])
    elif "energy" in chunk_or_batch_or_outputs:
        return len(chunk_or_batch_or_outputs["energy"])
    else:
        # todo: make this more generic
        raise KeyError
