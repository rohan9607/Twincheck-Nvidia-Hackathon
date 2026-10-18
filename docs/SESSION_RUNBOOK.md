# Real-arm session runbook

Three people, three jobs, the whole session: **Operator** (runs software), **Recorder** (ground truth),
**Camera** (films). Plant manager and Dr. Zeng present for anything that touches the arm.

## Day before

- [ ] Session time confirmed with the plant manager and Dr. Zeng
- [ ] Run sheet written: `ros2_ws/src/twincheck_bringup/config/run_sheet_<session>.yaml`
- [ ] Spark booked; `docker compose up -d` running; dashboard token on hand
- [ ] Detector exported **on the Spark**: `python perception/export_engine.py --weights <best.pt> --out models/overhead_obb.engine`
- [ ] Whole pipeline replayed on recorded real camera data (`ros2 bag play`) with no errors
- [ ] Position template printed (5 marked spots, `calib/template_spots.csv`)
- [ ] Event source decided with Dr. Zeng: `joint_states` (read-only) if allowed, else `manual` with `twincheck_trigger`

## Start of session (15 min)

```bash
# Spark, terminal 1: the pipeline (Watch mode, real arm)
ros2 launch twincheck_bringup twincheck.launch.py env:=real mode:=watch events:=joint_states \
    session_tag:=oct29_numbers run_sheet:=<run sheet> model_version:=<name>
# terminal 2 (manual events only): ros2 run twincheck_checker twincheck_trigger
# Recorder's laptop: python scripts/ground_truth.py --session oct29_numbers
```
1. Calibration check: object on each of the 5 spots → `python calib/verify.py ...` → mean/max error logged. Over 5 mm: recalibrate first.
2. Dashboard open on a second screen.

## Blocks (about 2 hours, 1 minute per trial)

| Block | Trials | Launch change | What happens |
|---|---|---|---|
| Baseline | 30 | `mode:=baseline` | Robot's own program; parts placed per run sheet, some shifted 2–4 cm |
| Watch, normal | 30 | `mode:=watch` | Same positions; TwinCheck checks every pick |
| Watch, induced failures | 3 × 10 | `mode:=watch` | Empty grasp, removed mid-grasp, nudged part (staff touch the table only with the arm paused, as the plant manager directs) |
| Guide (only if approved) | 30 | `mode:=guide armed:=true` | Vision-guided picks with retry; e-stop within reach |
| New object | 10–15 | `mode:=watch` | Object never used in development |
| Filming | — | — | Clearest caught failure from two angles, dashboard visible |

Keep settings fixed inside a block. The Recorder enters what a person saw, never what TwinCheck said.

## Before leaving the building

- [ ] All ground truth entered (`scripts/ground_truth.py` shows nothing pending)
- [ ] `python scripts/results.py --env real --session oct29_numbers`
- [ ] Surprises written down (they become the honest-limits slide)
- [ ] `pg_dump` and videos copied off the Spark
