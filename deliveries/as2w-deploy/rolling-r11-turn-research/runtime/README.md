# Runtime / output chain

## Action wrapper (version `r11-train-v1-fixdelay`)

```
actor raw(16)
  -> x6.0 clip (wrapper clip_actions)
  -> leg 0-11: x0.35 + default stand pose
     wheel 12-15: x16.0
  -> action-term clip (legs: 90% angle band; wheels: URDF 100%, non-binding)
  -> DelayedTarget (lag=0, pass-through; see action-delay-disable.patch)
  -> set_joint_position_target_index / set_joint_velocity_target_index
  -> PhysX implicit drive: legs kp=55/kd=2, wheels kp=0/kd=6, effort_limit=13.5 Nm
```

## 16-axis mapping (canonical order used everywhere)

`JOINTS = [f"{leg}_{part}" for part in ("hip","thigh","calf","foot") for leg in ("FL","FR","RL","RR")]`

| idx | joint | idx | joint | idx | joint | idx | joint |
|---|---|---|---|---|---|---|---|
| 0 | FL_hip | 4 | FL_thigh | 8 | FL_calf | 12 | FL_foot |
| 1 | FR_hip | 5 | FR_thigh | 9 | FR_calf | 13 | FR_foot |
| 2 | RL_hip | 6 | RL_thigh | 10 | RL_calf | 14 | RL_foot |
| 3 | RR_hip | 7 | RR_thigh | 11 | RR_calf | 15 | RR_foot |

MuJoCo actuator order is per-leg: canonical -> mujoco = [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], wheels at actuator [3,7,11,15].
Real-robot DDS slots (from the R4000 hardware contract, **different training line, hardware-side reference only**):
FL_hip 3 / FR_hip 0 / RL_hip 9 / RR_hip 6; thigh 4,1,10,7; calf 5,2,11,8; wheels FL 13 / FR 12 / RL 15 / RR 14; hip direction -1, others +1.

## Known caliber inconsistencies (documented, NOT changed)

1. wheel support reward uses world **z component + 2 N**; acceptance uses **|F| + 5 N**
2. rear thigh: hard termination 1.40 / soft penalty 1.30 / 90% angle band allows up to 3.851
3. knee: origin 45 mm / envelope 15 mm / soft margin 100 mm (two DIFFERENT geometric quantities; measured origin->surface offset about 36.6 mm)
4. non-wheel contact: training >4 N / MuJoCo eval any contact
