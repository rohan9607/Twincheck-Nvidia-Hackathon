"""Export trained weights to TensorRT on the machine that will run them (the Spark).

TensorRT engines are tied to the GPU they were built on, so always export on the
Spark itself, never copy an engine from Brev.

    python perception/export_engine.py --weights runs/obb/overhead_v2_n2000/weights/best.pt --out models/overhead_obb.engine
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO
    engine = YOLO(args.weights).export(format="engine", half=args.half, imgsz=args.imgsz)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(engine, args.out)
    shutil.copy(args.weights, Path(args.out).with_suffix(".pt"))
    print(f"engine -> {args.out} (and the .pt beside it, as a fallback)")


if __name__ == "__main__":
    main()
