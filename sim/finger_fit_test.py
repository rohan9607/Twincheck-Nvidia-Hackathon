"""Which objects fit between the standard fingers? (organizer feedback, point 1)

For every object prim in sim_config.yaml it measures the object's size from the
twin and compares its narrow side (what the fingers close across) with the jaw
opening from config/twincheck.yaml gripper.max_opening_mm.

    /isaac-sim/python.sh sim/finger_fit_test.py

Then confirm with real grasps in the twin: sim/twin_trial_runner.py --rung fixed_pose
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import ROOT, load_sim_config, start_app  # noqa: E402


def main():
    cfg = load_sim_config()
    import yaml
    proj = yaml.safe_load(open(ROOT / "config" / "twincheck.yaml"))
    jaw = proj["gripper"]["max_opening_mm"]
    clearance = proj["gripper"]["clearance_mm"]
    app = start_app(headless=True)
    from twin_common import open_stage, world_bbox_mm
    stage = open_stage(cfg["usd_path"])

    print(f"\njaw opening {jaw} mm, clearance {clearance} mm\n")
    print(f"{'object':<8} {'size x,y,z (mm)':<26} {'narrow side':>11}  fits?")
    for name, path in cfg["objects"].items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"{name:<8} prim not found at {path} (fix sim_config.yaml)")
            continue
        lo, hi = world_bbox_mm(stage, path)
        size = hi - lo
        narrow = float(min(size[0], size[1]))
        fits = narrow + clearance <= jaw
        print(f"{name:<8} {str(size.round(1).tolist()):<26} {narrow:>9.1f}mm  {'YES' if fits else 'NO  -> pick a smaller object or redesign fingers'}")
    print("\nCopy the sizes into config/twincheck.yaml (objects.*.width_mm / length_mm).")
    app.close()


if __name__ == "__main__":
    main()
