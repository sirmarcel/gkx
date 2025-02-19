from pathlib import Path
import numpy as np
from stepson import Trajectory

og = Trajectory("md/trajectory/")
re1 = Trajectory("md/recompute_serial/")

for i in range(int(len(og.time) / 2)):
    try:
        np.testing.assert_allclose(og.time.data[i * 2], re1.time.data[i])
        np.testing.assert_allclose(
            og.energy_potential.data[i * 2], re1.energy_potential.data[i], atol=1e-5
        )
        # concession to sad facts about numerics
        np.testing.assert_allclose(og.forces.data[i * 2], re1.forces.data[i], atol=1e-5)
    except AssertionError:
        print(i)
        raise

parallel = Path("md/recompute_parallel/")
if parallel.is_dir():
    re2 = Trajectory(parallel)

    for i in range(int(len(og.time) / 2)):
        np.testing.assert_allclose(og.time.data[i * 2], re2.time.data[i])
        np.testing.assert_allclose(
            og.energy_potential.data[i * 2], re2.energy_potential.data[i], atol=1e-5
        )
        # concession to sad facts about numerics
        np.testing.assert_allclose(og.forces.data[i * 2], re2.forces.data[i], atol=1e-5)
else:
    print("cannot test parallel!")
