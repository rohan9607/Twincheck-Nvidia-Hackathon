# Architecture

## Data flow for one pick

```
cameras ──images──> detector_node ──/perception/objects──┐
wrist cam ─────────> grasp_check_node ─/perception/grasp_state─┤
robot (joint_states | manual | guide) -> robot_state_node ─/robot/event─┤
                     expectation_node ─/twin/expected (config)─┤
     Isaac Sim shadow twin (optional) ─/twin/expected (twin)───┤
                                                               v
                                                      comparator_node ─/check/result─┬─> alert_node ─/alerts, /guide/command
                                                                                     ├─> resync_node ─/twin/set_poses ─> shadow twin
                                                                                     └─> trial_logger ─> Postgres ─> dashboard
```

## Topics

| Topic | Type | From → To |
|---|---|---|
| `/cam/{overhead_a,overhead_b,wrist}/image_raw` | sensor_msgs/Image | cameras or Isaac Sim → perception, comparator |
| `/perception/objects` | ObjectArray | detector → comparator, resync, guide |
| `/perception/grasp_state` | GraspState | grasp_check → comparator |
| `/robot/event` | RobotEvent | robot_state / guide → everyone |
| `/robot/manual_event` | std_msgs/String | twincheck_trigger or twin runner → robot_state |
| `/twin/expected` | ExpectedState | expectation_node, shadow twin → comparator |
| `/twin/objects`, `/twin/set_poses` | ObjectArray | shadow twin ↔ resync |
| `/twin/resync_report` | ResyncReport | resync → logger |
| `/check/result` | CheckResult | comparator → alert, resync, logger, guide |
| `/alerts`, `/guide/command` | std_msgs/String | alert → dashboard, guide |

## Frames and units

All poses are in the `table` frame in **millimeters and degrees**. The homography is fitted to markers whose
positions are read from the robot controller, so table x/y equals robot_base x/y. Heights are parameters.

## Timing

Every message carries the capture or controller timestamp. The comparator pairs each robot event with the
nearest frame within `sync_tolerance_ms` (100 ms) and logs the gap, like matching spans in a distributed trace.
Post-place waits `place_settle_ms` for the scene to settle. A missing wrist answer counts as a failure, never a pass.

## Modes

| | Baseline | Watch | Guide |
|---|---|---|---|
| Robot program | its own | its own | guide_node |
| Checks run | yes (logged, nobody acts) | yes | yes |
| On failure | nothing | alert | re-plan / retry once / stop |
| Needs controller commands | no | no | yes, plus mentor approval |

## Design choices

- **Check logic is pure Python** (`twincheck_core`), shared by ROS nodes, the offline simulator, demo data and
  analysis, so the numbers in the video come from the same code that ran on the arm.
- **Missed failures cost more than false alarms**: unknown grasp state fails; thresholds are chosen to catch
  every induced failure, then minimize false alarms (`scripts/results.py --sweep`).
- **Resync after every pick** keeps twin drift from piling up into false alarms; the shift is logged as a drift signal.
- **Guide is disarmed by default**, clamps every target to workspace limits and scales speed to 10%.
