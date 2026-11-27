import glob, time, csv, sys
import cv2
import numpy as np
from ultralytics import YOLO

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"   # "cpu" or "0"
half = len(sys.argv) > 2 and sys.argv[2] == "half"
tag = f"{'cpu' if device == 'cpu' else 'gpu'}_{'fp16' if half else 'fp32'}"
print(f"Running: {tag}")
kwargs = dict(device=device, verbose=False)
if half:
    kwargs["quantize"] = 16

model = YOLO("yolov8n.pt")

files = sorted(glob.glob("frames/*.png"))
files = [f for f in files if cv2.imread(f) is not None]  # skip corrupt images
print(f"{len(files)} valid frames")

# Warm-up so first-run overhead doesn't skew timings
for f in files[:10]:
    model(f, **kwargs)
e2e, infer, rows = [], [], []
for f in files:
    t0 = time.perf_counter()
    r = model(f, **kwargs)[0]
    dt = (time.perf_counter() - t0) * 1000
    e2e.append(dt)
    infer.append(r.speed["inference"])
    rows.append([f, len(r.boxes), round(dt, 2), round(r.speed["inference"], 2)])

with open(f"baseline_{tag}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "detections", "end_to_end_ms", "inference_ms"])
    w.writerows(rows)

print(f"end-to-end: mean {np.mean(e2e):.1f} ms ({1000/np.mean(e2e):.1f} FPS), p95 {np.percentile(e2e, 95):.1f} ms")
print(f"model only: mean {np.mean(infer):.1f} ms, p95 {np.percentile(infer, 95):.1f} ms")
print(f"total detections: {sum(r[1] for r in rows)}")