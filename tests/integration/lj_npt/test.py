from stepson import Trajectory

t = Trajectory("md/trajectory/")

print(t.pressure.compute())
print(t.volume.compute())
