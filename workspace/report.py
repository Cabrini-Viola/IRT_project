"""Report where the fingertips are relative to the cube and table, and suggest a shift."""
import sys
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

bodies = {}
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name:
        bodies[name] = data.xpos[i].copy()

tips = {n: p for n, p in bodies.items() if "distal" in n}
palm = next(p for n, p in bodies.items() if "palm" in n)
cube = bodies["object"]
mount = next(p for n, p in bodies.items() if "hand mount" in n)
tip_xy = np.mean([p[:2] for n, p in tips.items() if "thdistal" not in n], axis=0)

print(f"cube centre        : x={cube[0]:.3f} y={cube[1]:.3f} z={cube[2]:.3f}")
print(f"palm body          : x={palm[0]:.3f} y={palm[1]:.3f} z={palm[2]:.3f}")
print(f"mount              : x={mount[0]:.3f} y={mount[1]:.3f} z={mount[2]:.3f}")
for n, p in tips.items():
    print(f"{n:24s} height above table = {100 * p[2]:.1f} cm")
print(f"lowest fingertip   : {100 * min(p[2] for p in tips.values()):.1f} cm above the table")
dx, dy = (tip_xy - cube[:2]) * 100
print(f"mean fingertip is {dx:+.1f} cm in x and {dy:+.1f} cm in y from the cube centre")
new = mount[:2] - (tip_xy - cube[:2])
print(f'to centre the fingertips over the cube use: --mount-pos "{new[0]:.4f} {new[1]:.4f} {mount[2]:.4f}"')