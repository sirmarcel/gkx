import numpy as np

from gkx import comms


def digits(n):
    return int(np.ceil(np.log10(n)))


def print_diagnostics(md_state=None, system=None, calc_state=None, atoms=None):
    message = get_diagnostics(
        md_state=md_state, system=system, calc_state=calc_state, atoms=atoms
    )

    comms.state(message, title="glp diagnostics")


def get_diagnostics(md_state=None, system=None, calc_state=None, atoms=None):
    if md_state is not None:
        point = md_state.point
        calc_state = md_state.calc_state
        N = len(point.R)
    if atoms is not None:
        N = len(atoms.get_positions())

    else:
        if system is not None:
            N = len(system.R)
        elif atoms is not None:
            N = len(atoms)
            if calc_state is None:
                calc_state = atoms.calc.state

        # better fail here
        assert calc_state is not None

    message = [f"{N} atoms in simulation cell", ""]

    if hasattr(calc_state, "unfolding"):
        unfolding = calc_state.unfolding
        neighbors = calc_state.neighbors
    else:
        unfolding = None
        neighbors = calc_state

    # if unfolding is not None:
    #     Nadd = unfolding.padding_mask.sum()

    #     message.append("Unfolding:")
    #     message.append(f"{len(unfolding.padding_mask)} total positions in unfolded cell")
    #     message.append(f"  {Nadd} real replicas")
    #     message.append(f" +{N} original")
    #     message.append(f" ={N+Nadd} unfolded system size")
    #     message.append(f" +{(~unfolding.padding_mask).sum()} padding")

    if unfolding is not None:
        Nadd = len(unfolding.replica_idx)
        Nunf = N + Nadd
        d = digits(Nunf)

        message.append("Unfolding:")
        message.append(f"{Nunf:>{d+3}} total positions in unfolded cell")
        message.append(f"   {Nadd:>{d}} replica positions")
        message.append(f" + {N:>{d}} original positions")
        message.append(f" = {Nunf:>{d}} unfolded positions")
        message.append(f" + {0:>{d}} padding positions")
        message.append("")

    if neighbors is not None:
        mask = neighbors.centers != neighbors.reference_positions.shape[0]

        Npair = mask.sum()
        Npad = (~mask).sum()
        Ntol = len(mask)
        d = digits(Ntol)

        message.append("Neighborlist:")
        message.append(f"{Ntol:>{d+3}} total pairs")
        message.append(f"   {Npair:>{d}} real pairs")
        message.append(f" + {Npad:>{d}} padding pairs")
        message.append("")

        _, sizes = np.unique(neighbors.centers[mask], return_counts=True)
        max_nbh = sizes.max()
        min_nbh = sizes.min()
        mean_nbh = sizes.mean()
        d = digits(max_nbh)

        message.append("Neighbors:")
        message.append(f" {int(mean_nbh):>{d}} mean number of neighbors")
        message.append(f" {max_nbh:>{d}} max  number of neighbors")
        message.append(f" {min_nbh:>{d}} min  number of neighbors")

    message.append("")

    return message
