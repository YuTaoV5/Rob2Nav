# Exact Go2 MuJoCo model for unified deployment tests

`go2_isaac.xml` and its 16 `assets/*.obj` files are copied without modification from `GO2_backflip/mujoco/go2` at upstream commit `954aa61ceb62c515689677f44a21e32109f6b62d` (MIT license in `LICENSE_GO2_BACKFLIP`). The XML SHA256 is `fb81da82a8b5949bc0a45d5bf3a5003507c53d731cd43caf690cf729d05479d3`; individual hashes are in `SHA256SUMS`. These are the exact files used by the published free-base sideflip and gait/flip/gait verification. The visible meshes are required to compile the XML, although they do not supply the motor acceptance result.

This model is a simulation asset and a device-qualification input, not a verified representation of every physical Go2. Real deployment must match the calibrated machine's mass, COM, joints and firmware separately. The sideflip reference paired with this model is `../reference/sideflip.npz` (SHA256 `12877865c884becf477077a750fe1db3cdb01270c34a09ab35661d36ec375981`).
