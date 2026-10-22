# cell_policy: TwinCheck on the organizers' real cell

The real FANUC ER-4iA is run through the organizers' platform (hackathon.sidclab.org). Our machine runs a
`policy.py`; their server sends an observation (table, side and wrist images, joints, fingertip pose) and our
`act(obs)` returns the next fingertip targets. The cell plays them only during an approved 30-minute slot with an
operator present, and enforces its own limits. **This is the supervised way to command the real arm.**

| File | What it does |
|---|---|
| `twincheck_policy.py` | `Policy` for the client: runs `twincheck_core.PickLoop` (pick and place with the three checks), logs trials |
| `cell_perception.py` | Table-camera YOLO OBB → table mm with the organizers' calibration (`pixel_to_base`); wrist grasp check |
| `policy.py` | A 3-line shim: copy it over the client's `policy.py` |
| `make_placement.py` | Writes the client's `submission/placement.yaml` from `config/twincheck.yaml` (optionally nudged layouts) |

The pick loop, the checks and every limit live in `twincheck_core` (`pickloop.py`, `cell.py`) and are unit-tested.
Tunables are in `config/twincheck.yaml` under `cell:`.

## Set up (Linux, macOS, or Windows with Git Bash)

```bash
git clone https://github.com/SIDCLab/PhysicalAI-Hackathon-2026 && cd PhysicalAI-Hackathon-2026
uv sync && uv add pyyaml "psycopg[binary]"         # + `uv add ultralytics` once there is a detector model
cp ../twincheck/cell_policy/policy.py policy.py     # TWINCHECK_REPO=<repo> if the folders are not side by side
python ../twincheck/cell_policy/make_placement.py --out submission/placement.yaml
```

Then fill in `submission/submission.yaml` (team Wicked, `task_id` ending in our team letter, `model_type: Hybrid`).

## Run

```bash
export TWINCHECK_MODE=guide            # baseline | watch | guide
export TWINCHECK_SESSION=real_s1_guide # trial database tag; also runs/cell/<session>.jsonl
uv run run.py --mock                   # local mock: must end with "✓ Ready for the real robot"
uv run run.py --submit                 # then book a slot in the Calendar tab
uv run run.py                          # during the slot
```

On Windows set `PYTHONIOENCODING=utf-8`, or the mock's final "✓" line crashes the console print.
With `TWINCHECK_DSN` set (from `.env`), trials and checks go to Postgres and `scripts/results.py` reads them.
The operator scores each rollout; enter that ground truth with `scripts/ground_truth.py`.

## What each mode does on the cell

- **baseline**: a blind, taught program. It reaches for the `pick_points` and logs the checks without acting on them.
- **watch**: the same blind motion, but failed checks are alerts. This is the bolt-on monitor.
- **guide**: it aims at the part the table camera sees. A failed grasp is retried once from a fresh scan. A bad
  place, or a failure with no retries left, parks the arm and holds until the operator ends the rollout.

Compare them on the same nudged layouts (`make_placement.py --rollouts 10 --nudge-mm 30`, layout `different`).

## Verified so far (Oct 20)

- `uv run run.py --mock` gives "✓ Ready for the real robot" in all three modes, with 0 rejected actions.
- With no detector model and no wrist reference, every check fails safely (unknown = fail) and Guide stops.
- Trials and checks reach Postgres, and `results.py` reads the session.
- The mock has no kinematics or real images. Nothing here proves a grasp works: test in the starter twin first.

## Before a real slot (TODO)

- `cell.finger_yaw_offset_deg`: find the R at which the fingers close along base +x, in the twin. Screws need
  it; cubes do not, because they are 90-degree symmetric.
- `cell.park_tcp_mm`: check that this pose keeps the arm out of the table camera's view of the parts and bins.
- A detector trained on twin renders of the **10 mm cubes and M6 screws** (`models.detector`), and a wrist
  reference image of the empty closed gripper (`models.wrist_reference`) or another wrist method.
- The placement file gives a screw's position as its **head**; we grasp the shank centre. Check where the
  operator puts it and offset `pick_points` if needed.
- The real arm follows its targets about 0.8 s behind (the twin does not). `settle_rows` holds each stop for
  0.8 s; increase it if the wrist or table images show the arm still moving.
