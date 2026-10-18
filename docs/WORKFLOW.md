# Workflow: what to run, in order

Each step has an owner, the exact command, and a "done when". Don't skip a "done when".

## Week of Oct 17: set up and see the twin

| # | Owner | Do | Done when |
|---|---|---|---|
| 1 | All | [SETUP](SETUP.md) steps 1–3; `make test`, `make simulate` | Everyone can run the simulation |
| 2 | Captain | Email Dr. Zeng: controller options, read-only robot state, CSI visit, two arm sessions | Sent |
| 3 | Code | `sim/explore_twin.py` → fill `sim/sim_config.yaml` + object sizes | Report shows cameras, joints, objects |
| 4 | Code | `sim/finger_fit_test.py`; pick objects that fit; update `config/twincheck.yaml` | Scenario objects decided |
| 5 | Data | `python calib/make_targets.py`; print; practice `calib/intrinsics.py --webcam 0` | Reprojection error < 0.5 px on a webcam |
| 6 | Story | Scenario + business case draft; trial-log sheet; run sheets for each session | Drafts shared |

## Oct 19–22: synthetic data, detector, first loop in the twin

| # | Owner | Do | Done when |
|---|---|---|---|
| 7 | Code | `sim/replicator_overhead.py --n 500 --seed 42 --out data/overhead_v1_seed42_n500` | 500 images |
| 8 | Data | `perception/draw_labels.py --data data/overhead_v1_seed42_n500 --n 20 --out data/check_v1` | 20 images checked by eye, boxes tight |
| 9 | Code | Render the main set: `--n 4000 --seed 7 --out data/overhead_v2_seed7_n4000` | Rendered once; copied to team drive |
| 10 | Data | `perception/train.py obb --data data/overhead_v2_seed7_n4000/data.yaml --subsets 1000 2000 4000 --real data/real_test_v1` | Learning curve logged |
| 11 | Code | `sim/replicator_wrist.py --n 1500 --out data/wrist_v1_seed3_n1500`; copy `empty_reference.png` to `models/wrist_empty_ref.png` | Wrist data + reference |
| 12 | Code | `make ros`; run the twin loop (below) | **Gate G2: the twin flags a moved object and logs a trial** |

Twin loop (three terminals, same `ROS_DOMAIN_ID`):
```bash
ros2 launch twincheck_bringup twincheck.launch.py env:=twin mode:=watch events:=manual session_tag:=twin_g2
/isaac-sim/python.sh sim/twin_runtime.py --role trials --rung fixed_pose --n 20 --induce nudged_part
python scripts/results.py --env twin
```

## Oct 21–23: CSI visit, calibration, real test sets

| # | Owner | Do | Done when |
|---|---|---|---|
| 13 | All | At CSI: checkerboard photos per camera → `calib/intrinsics.py` | K saved for 3 cameras |
| 14 | All (supervised) | Markers + arm tool tip → `calib/marker_points.csv` → `calib/homography.py` per overhead camera | Leave-one-out error < 5 mm |
| 15 | Data | Shoot 100–200 overhead + 30–50 wrist test photos; label in CVAT (YOLO OBB export) → `data/real_test_v1` | Labeled, never used for training |
| 16 | Code | `calib/twin_camera_match.py --print-usd-lens` → set twin cameras; `--overlay` check | Twin edges sit on real edges |
| 17 | Data | `perception/eval_real.py` on the best model | **Gate G3: recall ≥ 95%, median ≤ 5 mm, ≤ 5°** |
| 18 | Data | `perception/eval_wrist.py` for reference_diff, then cosmos (`perception/cosmos_probe.py` first) | Method chosen; zero empty-called-held |

If G3 fails: widen randomization (`--lighting wide`, more camera jitter, clutter), re-render once, retrain once.

## Oct 22–25: full checks, ladder, threshold, Guide in the twin

| # | Owner | Do | Done when |
|---|---|---|---|
| 19 | Code | Twin trials per rung: fixed_pose → random_pos → random_rot → second_object → clutter → lighting; 100 each, baseline and watch | Logged per rung |
| 20 | Code | Induced failures: 100 each of `empty_grasp`, `removed_mid_grasp`, `nudged_part` | Logged |
| 21 | Data | `scripts/results.py --sweep --env twin` → set `thresholds.pre_pick_mm` | Threshold chosen from data |
| 22 | Code | Guide in the twin, disarmed then sim-armed: `mode:=guide` | Retry works in the twin |
| 23 | Code | Shadow twin dry run against recorded real data: `sim/twin_runtime.py --role shadow` | Resync shifts logged |
| 24 | Story | Dashboard on the Spark (`docker compose up -d`), results charts, business case numbers | Charts update from the DB |

## Oct 26–29: real arm → [SESSION_RUNBOOK](SESSION_RUNBOOK.md)

Session 1 (dry run, gate G4) and session 2 (numbers, gate G5). No new features after Oct 29.

## Oct 30–Nov 1: video, summary, submit

`python scripts/results.py --csv runs/results_final.csv` gives every number. Video: problem, a caught failure
live, numbers vs baseline, twin vs real, honest limits. Submit by 11:59 PM CT Nov 1.
Then `pg_dump` the database and copy `runs/`, `data/`, `models/` off the Spark.
