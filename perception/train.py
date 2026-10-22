"""Train the overhead detector or the wrist classifier, then log the run.

    # overhead oriented-box detector (start with nano)
    python perception/train.py obb --data data/overhead_v1_seed42_n1000/data.yaml --real data/real_test_v1

    # learning curve: 1k, 2k, 4k images, each scored on the real test set
    python perception/train.py obb --data data/overhead_v2_seed7_n4000/data.yaml --subsets 1000 2000 4000 --real data/real_test_v1

    # wrist classifier (only if reference_diff and Cosmos are not good enough)
    python perception/train.py cls --data data/wrist_v1_seed3_n1500

Every run is written to the training_runs table (if TWINCHECK_DSN is reachable)
and to runs/training_log.jsonl, with GPU, minutes, dataset and scores.
Brev users: add --credits-per-hour so credits are recorded too, and STOP the instance after.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpu_name() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception:
        return "cpu"


def log_run(row: dict):
    log = ROOT / "runs" / "training_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")
    try:
        import psycopg
        with psycopg.connect(os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck"),
                             connect_timeout=3, autocommit=True) as db:
            db.execute("""INSERT INTO training_runs (model, dataset, epochs, gpu, runtime_min, credits_used, synth_score,
                          real_recall, real_pos_err_mm, real_yaw_err_deg, weights_path, notes)
                          VALUES (%(model)s,%(dataset)s,%(epochs)s,%(gpu)s,%(runtime_min)s,%(credits_used)s,%(synth_score)s,
                          %(real_recall)s,%(real_pos_err_mm)s,%(real_yaw_err_deg)s,%(weights_path)s,%(notes)s)""", row)
    except Exception as e:
        print(f"(database not reachable, logged to {log} only: {e})")


def subset_dataset(data_yaml: Path, n: int, seed: int) -> Path:
    """Make a smaller copy (symlinks) of a YOLO dataset with n training images."""
    import yaml
    d = yaml.safe_load(open(data_yaml))
    root = Path(d["path"])
    imgs = sorted((root / d["train"]).glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    sub = root.parent / f"{root.name}_sub{n}"
    if sub.exists():
        shutil.rmtree(sub)
    for split, files in (("train", imgs[:n]), ("val", sorted((root / d["val"]).glob("*.jpg")))):
        (sub / "images" / split).mkdir(parents=True)
        (sub / "labels" / split).mkdir(parents=True)
        for f in files:
            os.symlink(f.resolve(), sub / "images" / split / f.name)
            lab = root / "labels" / split / (f.stem + ".txt")
            if lab.exists():
                os.symlink(lab.resolve(), sub / "labels" / split / lab.name)
    names = "".join(f"  {k}: {v}\n" for k, v in d["names"].items())
    (sub / "data.yaml").write_text(f"path: {sub}\ntrain: images/train\nval: images/val\nnames:\n{names}")
    return sub / "data.yaml"


def train_one(kind, data, model, epochs, imgsz, batch, seed, name, workers, cache):
    from ultralytics import YOLO
    m = YOLO(model)
    t0 = time.time()
    res = m.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, project=str(ROOT / "runs" / kind),
                  name=name, exist_ok=False, patience=20, verbose=False, workers=workers, cache=cache)
    minutes = (time.time() - t0) / 60
    weights = Path(res.save_dir) / "weights" / "best.pt"
    metrics = getattr(res, "results_dict", {}) or {}
    score = metrics.get("metrics/mAP50(B)") or metrics.get("metrics/accuracy_top1")
    return weights, minutes, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["obb", "cls"])
    ap.add_argument("--data", required=True, help="obb: data.yaml; cls: dataset folder with train/ val/")
    ap.add_argument("--model", help="default yolo11n-obb.pt / yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsets", type=int, nargs="*", help="learning curve: training-set sizes")
    ap.add_argument("--real", help="real test set folder (obb) for eval_real.py after each run")
    ap.add_argument("--camera", default="table")
    ap.add_argument("--credits-per-hour", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8,
                    help="data-loader processes; 0 in containers with a small /dev/shm (the YAAD Spark has 64 MB)")
    ap.add_argument("--cache", choices=["ram", "disk"], default=None,
                    help="keep decoded images (ram: fastest; pairs well with --workers 0)")
    args = ap.parse_args()

    model = args.model or ("yolo11n-obb.pt" if args.kind == "obb" else "yolo11n-cls.pt")
    imgsz = args.imgsz or (640 if args.kind == "obb" else 224)
    sizes = args.subsets or [None]
    gpu = gpu_name()
    print(f"GPU: {gpu}")
    for n in sizes:
        data = subset_dataset(Path(args.data), n, args.seed) if (n and args.kind == "obb") else Path(args.data)
        name = f"{(Path(args.data).parent.name or Path(args.data).stem) if args.kind == 'obb' else Path(args.data).name}" + (f"_n{n}" if n else "")
        weights, minutes, score = train_one(args.kind, data, model, args.epochs, imgsz, args.batch, args.seed, name, args.workers, args.cache)
        row = {"model": Path(model).stem, "dataset": name, "epochs": args.epochs, "gpu": gpu,
               "runtime_min": round(minutes, 1), "credits_used": round(minutes / 60 * args.credits_per_hour, 2),
               "synth_score": score, "real_recall": None, "real_pos_err_mm": None, "real_yaw_err_deg": None,
               "weights_path": str(weights), "notes": ""}
        if args.real and args.kind == "obb":
            out = ROOT / "runs" / "eval" / name
            subprocess.run([sys.executable, str(ROOT / "perception" / "eval_real.py"), "--model", str(weights),
                            "--data", args.real, "--camera", args.camera, "--out", str(out)], check=False)
            sfile = out / "summary.json"
            if sfile.exists():
                s = json.loads(sfile.read_text())
                row.update(real_recall=s["recall"], real_pos_err_mm=s["median_pos_mm"], real_yaw_err_deg=s["median_yaw_deg"])
        log_run(row)
        print(f"\n== {name}: synthetic {score}, real recall {row['real_recall']}, pos {row['real_pos_err_mm']} mm, "
              f"{minutes:.0f} min on {gpu}\n")
    if len(sizes) > 1:
        print("Learning curve done. If real recall stopped rising, stop rendering more images; "
              "change the randomization instead (lighting, textures, camera jitter).")


if __name__ == "__main__":
    main()
