# Jenga-playing robot simulation

This project implements the robotics pipeline described in the presentation
using PyBullet's Franka Panda and an 18-layer, 54-block Jenga tower:

- an eye-in-hand overview scan that provides image-confirmed candidates before
  the block-selection policy chooses a target;
- simulator segmentation as a ground-truth perception baseline and an OpenCV
  color/depth candidate detector with target pose estimation for comparison;
- a two-subspace block policy with per-block state and one extraction per level
  (the current Panda/camera setup uses validated layers 4 through 15);
- Panda inverse kinematics and a trapezoidal joint-space approach trajectory;
- eye-in-hand RGB-D feedback, Jacobian differential kinematics, and a 7 mm
  contact-alignment tolerance;
- 240 Hz simulated contact-force feedback during visually guided pushing, with
  force traces logged at 20 Hz, a 0.23 N target, and the presentation's 0.32 N
  safety cutoff;
- open-loop/closed-loop comparisons and per-trial CSV, JSON, and plot outputs.

The overview scan evaluates all present block faces from image color and depth
before the policy selects among visible, eligible blocks. The OpenCV path uses
the known tower grid as a data-association prior, but does not receive a
preselected target block: it checks the image color and depth at every
projected candidate face, then the policy chooses among image-confirmed
candidates. It remains a map-assisted classical vision baseline, not a trained
YOLACT++ model. Likewise, the original paper's ViSP model-based tracker and
physical e.DO setup are not included; this implementation targets the
presentation pipeline on the available Panda/PyBullet assets. During pushing,
visual updates keep the end effector aligned when the block is visible; when
the probe occludes it, the last valid image estimate is held while the force
loop governs push speed. The planned push continues through temporary loss of
visual contact and stops on extraction clearance, the travel limit, or the force
safety threshold. The policy excludes layer 16 because repeatability checks
found that selecting this second-to-top layer can destabilize the upper tower.

## Environment setup

Use an Anaconda or Miniconda installation with Conda available in Anaconda
Prompt or a Conda-enabled PowerShell terminal. Python 3.10 is the tested
version. From the repository directory, create and activate an environment,
then install the runtime libraries:

```powershell
conda create -n jenga-robot python=3.10 pip -y
conda activate jenga-robot
python -m pip install pybullet numpy opencv-python matplotlib
```

The project uses PyBullet (including its `pybullet_data` assets), NumPy,
OpenCV's Python package (`opencv-python`, imported as `cv2`), and Matplotlib.
No separate installation is needed for `pybullet_data`; it is included with
PyBullet. The Python standard library and `unittest` are used for other
functionality and require no additional packages.

Verify the installation by running the tests:

```powershell
python -m unittest -v
```

If you already have the project's Anaconda `pybullet` environment, you can
reuse it instead of creating a new one by running `conda activate pybullet`.
Run the remaining commands from the activated environment.

Run one headless game with simulator ground-truth perception and closed-loop
control:

```powershell
python .\jenga_replication.py --detector ground_truth --controller closed_loop --trials 1 --output results
```

Run the 18-trial evaluation requested in the presentation:

```powershell
python .\jenga_replication.py --compare --trials 18 --output results
```

`--compare` evaluates both detectors against both open-loop and closed-loop
controllers. Add `--gui` for visualization. Each invocation writes
`attempts.csv`, `summary.json`, `outcomes.png`, `force_profiles.png`, and
`trajectories.png` under the selected output directory.

## Physical parameter caveat

The presentation-based model uses 75 x 25 x 15 mm blocks. The supplied
mechanics paper reports a sample-average mass of 19.6 g, average dimensions of
81 x 26 x 18 mm, and a typical hardwood static-friction coefficient of 0.4.
The default simulation now uses the paper's 19.6 g and a fixed 0.4 PyBullet
lateral-friction coefficient while retaining the presentation's dimensions;
the dimensions therefore remain a deliberate, documented source difference.
PyBullet's contact friction is only an approximation to the paper's static
Coulomb model, not a calibrated static/kinetic wood model. The values are
exposed as `--block-mass-kg`, `--friction-min`, and `--friction-max`, and the
generated summary records them as assumptions.

The mechanics paper estimates about 0.23 N to initiate sliding in its
simplified model. The force controller uses 0.23 N as its target and proportionally adjusts
push-axis velocity at the 240 Hz simulation rate using the simulated contact
force; force traces are recorded at 20 Hz. Axial speed is capped at 0.005 m/s,
and the push stops at the presentation's 0.32 N safety cutoff. The
presentation's 0.18 N and 0.32 N values also remain the policy's
classification thresholds. Rigid simulated contact can still produce peaks
above the cutoff within a physics step, so this is a simulated safety response
rather than a validated force-limiting guarantee.
Outcomes use measured axial travel and vertical displacement; the robot stops
as soon as the block clears its layer. Results are simulation outcomes, not
physical calibration or experimental replication scores.
