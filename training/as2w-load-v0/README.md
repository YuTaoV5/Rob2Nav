# AS2W load-v0 training asset

`load-v0` is the experimental longitudinal-load asset shared by the
Unitree-style and W1W HIMLoco AS2W retraining tasks.

| URDF parameter | Vendor asset | load-v0 |
| --- | ---: | ---: |
| `base_link` mass | 10.080 kg | 11.592 kg |
| Whole model mass | 26.000 kg | 27.512 kg |
| `base_link` COM X | +35.002 mm | +20.002 mm |
| `base_link` inertia tensor at its COM | Vendor tensor | Vendor tensor × 1.15 |

The change is confined to the `base_link` inertial block. The COM Y/Z
coordinates, all other links, joint axes and wheel collision meshes use the
vendor values. The mass and COM changes are experimental assumptions.

## Generate the exact training URDF

Use the AS2W description from [Unitree ROS at commit 157ff95](https://github.com/unitreerobotics/unitree_ros/tree/157ff950351e5f535e03c5d85a16b7d594b3dbcc/robots/as2w_description).
Keep the vendor `meshes/` alongside its `as2w.urdf`. From this repository:

```sh
python3 training/as2w-load-v0/generate.py /path/to/as2w_description/as2w.urdf
```

The script checks both SHA-256 sums and writes a new `as2w-load-v0.urdf`
next to the input so both descriptions resolve the same mesh paths. The
generated training asset has SHA-256
`7c9740a95042bad7b73cadfda90bdc7c1873ac766afbb7c0e7e492367d45afb0`.

## Verification and limits

A PhysX fixed-position hold covered T1 and T2 at 55/2 and 85/3.2, with two
initial perturbations per case. All eight runs had four loaded wheels and
zero mean non-wheel contact. At T2, 55/2, the rear calf deflection rose
from about 0.17–0.22 rad on the vendor asset to 0.23–0.25 rad on load-v0.

The hip-load mismatch remains unresolved. Separate MuJoCo command replays
showed no improvement in per-axis angle error. Treat load-v0 as a retraining
comparison, with independent checkpoints for the two tasks. It has not
passed field calibration or deployment validation.
