# TwinCheck

**Team Wicked · Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 3: Grasping and Sorting**

A fixed-motion FANUC arm is blind: if a part is a few centimeters off, it grabs air and nobody notices.
TwinCheck gives it eyes and a checker. Cameras watch every pick, the digital twin says what *should*
have happened, and when the two disagree TwinCheck raises an alert (Watch mode) or corrects and
retries (Guide mode). The detector is trained only on images rendered by the twin.

On every pick, three checks:

| When | Question | Evidence |
|---|---|---|
| Before the grab | Is the object where the robot will grab? | Overhead cameras |
| After the grab | Is the object in the gripper? | Wrist camera (the gripper has no part sensor) |
| After the drop | Did it land in the right bin? | Overhead cameras |

## What's in the repo

```
config/twincheck.yaml   ONE config: objects, bins, pick points, thresholds, cameras, safety limits
twincheck_core/         Pure-Python check logic (geometry, checks, comparator, resync, stats) + tests
ros2_ws/src/            ROS 2 Jazzy packages
  twincheck_msgs          messages (objects, grasp state, robot events, expectations, check results)
  twincheck_perception    detector_node (2 overhead cams, fused), grasp_check_node (wrist cam)
  twincheck_checker       robot_state_node, expectation_node, comparator_node, resync_node, alert_node
  twincheck_logger        trial_logger -> Postgres
  twincheck_guide         guide_node: vision-guided pick + retry (DISARMED by default)
  twincheck_bringup       one launch file for every mode
sim/                    Isaac Sim scripts: explore twin, finger fit, synthetic data, twin trials, shadow twin
calib/                  print targets, lens calibration, pixel->robot homography, verification, twin camera match
perception/             label check, training (+learning curve, run logging), real-image eval, wrist eval, Cosmos probe
scripts/                results tables, ground-truth entry, compute/credit log, offline simulation, demo data
dashboard/              FastAPI + React operator view (live alerts with expected vs observed images, results)
db/schema.sql           trials, checks, training_runs, compute_sessions, trial_summary view
docker/, docker-compose.yml, Makefile, .github/workflows/ci.yml
docs/                   SETUP, WORKFLOW (what to run, in order), SESSION_RUNBOOK (real arm), ARCHITECTURE
```

## Start here

1. **Today, any laptop, no ROS needed:**
   ```bash
   git clone <team repo> && cd twincheck
   python3 -m venv .venv && source .venv/bin/activate && pip install -e twincheck_core pytest
   make test          # 22 tests of the check logic
   make simulate      # 200 fake trials through the real comparator
   ```
2. **Set up machines:** [docs/SETUP.md](docs/SETUP.md) (VPN, YAAD Spark, ROS 2, database, Brev).
3. **Do the work in order:** [docs/WORKFLOW.md](docs/WORKFLOW.md), each step names the exact command and its "done when".
4. **Real-arm sessions:** [docs/SESSION_RUNBOOK.md](docs/SESSION_RUNBOOK.md).

## Status of each part (be honest in the video too)

| Part | State |
|---|---|
| Check logic, comparator, resync, stats | Done, unit tested (`make test`) |
| Database, results tables, dashboard | Done, tested against Postgres with demo data |
| ROS 2 nodes and launch | Written; build is checked in CI (`ros2-build`); first live run needs the twin |
| Isaac Sim scripts | Written for Isaac Sim 4.5/5.x; prim paths are TODO until `sim/explore_twin.py` runs on the real twin |
| Calibration and perception tools | Done; need real images from CSI |
| Guide mode | Written, disarmed by default; needs Dr. Zeng's confirmation of controller options, MoveIt group, gripper I/O |

Every `TODO` in `config/twincheck.yaml` and `sim/sim_config.yaml` is a placeholder to replace with a measured value.

## Safety rules (non-negotiable)

- No command to the physical arm and no controller change without the plant manager and Dr. Zeng (assigned mentor) present.
- Guide mode starts disarmed. `armed:=true` only during a supervised session.
- No printed finger on the arm before its clearance review.
- Real line data is for testing and looking, never for training.
- Never commit `.env`, keys, passwords or tokens. Put a token on every open port (the dashboard already requires one).

## Contacts

Robot arm and Spark machines: Dr. Zeng (zhenzeng@uwm.edu) · Digital twin: Dr. Riley (driley@nvidia.com) ·
CSI visits, VPN, physical tests: Shamar Webster (webste63@uwm.edu, `#data-and-testbed-requests`) ·
Brev: Dan Schneider (`#compute-questions`).
