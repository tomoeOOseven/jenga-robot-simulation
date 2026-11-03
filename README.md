# Jenga-playing robot: simulated replication in PyBullet

A PyBullet replication of Marchionna et al. (2023), *Deep Instance Segmentation
and Visual Servoing to Play Jenga with a Cost-Effective Robotic System*
(`2211.07977v2.pdf`), following the AR523 presentation. A 7-DOF Franka Panda
with an eye-in-hand RGB-D camera and a pushing finger plays an 18-level,
54-block tower.

## Pipeline

| Stage | Paper / presentation | This implementation |
|---|---|---|
| 1. Detection | YOLACT++ masks, PnP, ViSP tracker | `ground_truth`: perfect simulator masks and poses. `opencv`: colour threshold, depth edges and contours, PnP on face corners, depth refinement, then association with a tower map registered from the image. The OpenCV path never reads simulator poses. |
| 2. Selection | Two sub-spaces, random start, memory buffer, one extraction per level | Same. 14 levels (2–15), so the paper's minimum of 42 attempts holds. The first sub-space uses 0.32 N; once it is fully tested the policy switches sub-space and uses 0.18 N. After an immobile block it tries the rest of that level, centre block first. |
| 3. Approach | IK + trapezoidal trajectory | Damped least-squares IK on the controller's kinematic model, a trapezoidal joint-space move to a staging pose, then a straight-line approach |
| 4. Alignment and push | 2½-D visual servo `v = -λ L⁺ e`, s = (x, y, log Z/Z*, θu), ~9 Hz; open-loop push with a 1-D force sensor at 20 Hz | Same control law and rates. The camera twist is mapped to joints with the Jacobian pseudo-inverse. The push runs at constant speed and stops when the tared, 12-bit sensor reading exceeds the phase threshold (immobile) or the block clears the tower (extracted). |

The threshold is a classification test. It applies until the block has slid
15 mm, which proves it is mobile. After that the push continues under a 0.64 N
safety limit. Near the end of an extraction the block tips against the level
above and friction rises, so applying the threshold all the way would wrongly
stop pushes on blocks that are already almost out. An "immobile" call only
counts as correct if the finger was touching the target block alone; otherwise
(for example, a misaligned finger pressing on the level above) it is scored as
an error.

The 1-D force sensor is a spring-loaded fingertip on a prismatic joint
(100 N/m), so its readings come from the simulated contact. It only senses
force along the finger, like the paper's FMA sensor.

### Modelled error sources

Simulation is too perfect by default, so these real-world effects are modelled:

- **Block tolerances**: each block's height varies (σ = 0.1 mm), plus ±0.5 mm
  placement jitter. Different towers then have different friction patterns,
  which is why the paper says vision alone cannot classify removability.
- **Robot calibration error**: the controller reads joint encoders with a fixed
  per-joint offset (σ = 0.17°). Its kinematic model, IK and camera pose are
  slightly wrong; images are always rendered from the true camera pose.
- **Depth sensor**: per-pixel noise and a ±1 % systematic scale error per camera.

The closed-loop servo measures the block relative to the camera, so it is
largely immune to the first two. The open-loop (IK-only) baseline plans from
the overview estimate and is not.

## Environment setup

Use Anaconda or Miniconda (Python 3.10 tested):

```powershell
conda create -n jenga-robot python=3.10 pip -y
conda activate jenga-robot
python -m pip install pybullet numpy opencv-python matplotlib
```

You can reuse an existing `pybullet` environment with `conda activate pybullet`.
Run the tests (about 3 minutes; the `SimulationTests` class runs PyBullet):

```powershell
python -m unittest -v
```

## Running

One game, ground-truth detection, closed loop:

```powershell
python .\jenga_replication.py --trials 1 --output results
```

The presentation's evaluation: every detector × controller pair on the same
towers (trial *i* uses seed *i* for all four, so results are paired):

```powershell
python .\jenga_replication.py --compare --trials 18 --output results
```

Games run in parallel (`--workers`, default CPU count − 1). A game of about 40
attempts takes roughly 10–15 minutes on one core, so the full 72-game run takes
a few hours. Results are rewritten after every finished game, so you can look
at partial results during a long run, and a game that crashes is reported
without stopping the others. Add `--gui` to watch a game (this forces one
worker).

Useful options: `--max-attempts`, `--seed`, `--block-mass-kg`,
`--friction-min/--friction-max`, `--height-tolerance-mm`, `--joint-bias-deg`
(increase it to stress the open-loop baseline).

### Outputs

- `attempts.csv`: one row per attempt (level, slot, phase, threshold, outcome,
  failure reason, peak force, alignment error h/v, servo time, block
  displacement, tower perturbation, collapse).
- `summary.json`: assumptions, configuration, and per-combination metrics with
  mean ± std and 95 % t-intervals. Metrics include successful attempts per trial
  (the paper's primary metric), extraction / immobile / error rates, failure
  reasons, alignment error, fraction within 7 mm, servo time per level, and
  peak force by outcome.
- `outcomes.png` (cf. paper Fig. 10), `force_profiles.png` (Fig. 6),
  `alignment_and_servo.png` (Figs. 8 and 9), and `trajectories.png`.

An attempt is **successful** if the block is extracted, or correctly found
immobile, without collapsing the tower (paper section 4.5). The tower counts as
collapsed when any other block falls: it drops, tilts more than 20°, or slides
half a block length. Smaller drags of nearby blocks are reported as tower
perturbation. A target that tips out of the tower before the finger finishes
the push counts as extracted, since it has left the tower. Errors are
unreachable poses, lost tracking, servo timeouts, and pushes that do not clear
the block (for example, a side block that yaws off the finger).

## Physical parameters and limitations

- Blocks are 75 × 25 × 15 mm (presentation and paper). Mass is 19.6 g and
  friction μ = 0.4 (Ziglar 2006, `Analysis_of_Mechanics_in_Jenga.pdf`). None of
  these are calibrated to a real set.
- PyBullet's iterative solver under-resolves friction in a tall stack.
  Breakaway forces keep rising with solver iterations, so 300 iterations are
  used. Forces fall in the paper's 0–1 N range (lower levels about 0.5 N, upper
  levels about 0.05 N) but are not exact Coulomb values.
- Resting blocks sleep and use anchored friction. This stops the slow creep that
  stacked boxes otherwise show in Bullet.
- An extracted block is removed once it fully leaves the tower. In the paper it
  falls behind the tower, out of the robot's reach.
- The OpenCV detector is a classical baseline, not a trained YOLACT++ model.
  Block faces are drawn with dark seams (like chamfered edges) so touching faces
  can be separated. The ViSP tracker is replaced by per-frame detection plus
  data association.
- Blocks are not re-placed on top (as in the paper's robotic setting), and the
  tower is not repositioned between sub-spaces; the Panda reaches every level.
- Results are simulation outcomes, not a physical replication.
