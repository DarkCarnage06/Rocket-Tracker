# tools/adapter_3cam.py
"""
Adapter: run detection on 3 video sources (or use detection CSVs), synchronize detections,
triangulate using tools/triangulation.py and save results (CSV + plot).

Usage examples:
  python tools/adapter_3cam.py --cams cam1.mp4 cam2.mp4 cam3.mp4 --calibs calib1.npz calib2.npz calib3.npz --out results.csv
"""

import argparse
import time
import csv
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# import your repo's detector API here.
# Example: from rocket_tracker.detector import RocketDetector
# For generic fallback, adapter can accept precomputed detection CSVs.
try:
    from rocket_tracker.detector import RocketDetector  # <- change to actual module in repo
    HAVE_DETECTOR = True
except Exception:
    HAVE_DETECTOR = False

from triangulation import triangulate_n_cameras
import cv2
import numpy as np

def load_calib_npz(path):
    d = np.load(path, allow_pickle=True)
    K = d["K"]
    dist = d["dist"]
    # try to read R_world (cam->world) and C center; if not present, expect stereo extrinsics in another place
    R_world = d.get("R_world", None)
    C = d.get("C", None)
    if R_world is None:
        # fallback: if user saved rvec/tvec from calibrateCamera (rvec maps world->cam)
        rvec = d.get("rvec", None)
        tvec = d.get("tvec", None)
        if rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            # assume rvec/tvec map world->cam; camera center = -R.T @ t
            C = -R.T.dot(tvec.reshape(3,))
            R_world = R.T
    return {"K":K, "dist":dist, "R":R_world, "C":C}

def centroid_from_bbox(bbox):
    # bbox = (x_min,y_min,x_max,y_max) or (x,y,w,h) - handle both
    if len(bbox)==4:
        x0,y0,x1,y1 = bbox
        cx = (x0+x1)/2.0
        cy = (y0+y1)/2.0
        return (cx,cy)
    else:
        return None

def run_adapter(video_paths, calib_paths, out_csv, detection_csvs=None, max_dt=0.02):
    """
    If detection_csvs is None: will try to use repo detector on each frame.
    detection_csvs: optional list of CSVs per camera: timestamp, cx, cy
    """
    # load calibrations
    cams = [load_calib_npz(p) for p in calib_paths]
    Ks = [c["K"] for c in cams]
    dists = [c["dist"] for c in cams]
    Rs = [c["R"] for c in cams]
    Cs = [c["C"] for c in cams]

    # open video captures
    caps = [cv2.VideoCapture(p) for p in video_paths]
    fpss = [cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30 for cap in caps]

    # Prepare detector objects or load CSVs
    if detection_csvs:
        # load into deques keyed by timestamp
        det_bufs = []
        for csvp in detection_csvs:
            dq = deque()
            with open(csvp, "r") as fh:
                rdr = csv.reader(fh)
                for row in rdr:
                    ts = float(row[0]); cx=float(row[1]); cy=float(row[2])
                    dq.append((ts, (cx,cy)))
            det_bufs.append(dq)
    else:
        det_bufs = [deque() for _ in range(3)]
        if not HAVE_DETECTOR:
            raise RuntimeError("Repo detector not importable — either pass detection CSVs or adapt the import path in this script.")
        detectors = [RocketDetector() for _ in range(3)]

    timestamps = []
    distances = []
    residuals = []

    start_time = time.time()
    frame_idx = 0
    # We'll timestamp frames by elapsed time from start_time (for prerecorded videos, better to use frame index/fps)
    while True:
        frames = []
        rets = []
        tnow = time.time() - start_time
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            rets.append(ret)
            frames.append(frame)
        if not all(rets):
            break

        # detect in frames (if no CSVs)
        if not detection_csvs:
            for i, frame in enumerate(frames):
                # call the repo detector API
                dets = detectors[i].detect_frame(frame)  # adapt to the exact API
                # dets: list of bboxes [x,y,w,h] or [x1,y1,x2,y2]
                if dets:
                    # take the first / highest score detection
                    bbox = dets[0]["bbox"] if isinstance(dets[0], dict) and "bbox" in dets[0] else dets[0]
                    cx, cy = centroid_from_bbox(bbox)
                    if cx is not None:
                        det_bufs[i].append((tnow, (cx,cy)))
        # else CSVs already loaded

        # Try to find matching triplet within max_dt (simple greedy)
        # find earliest time in buf0 that has matching near times in buf1 and buf2
        if len(det_bufs[0]) and len(det_bufs[1]) and len(det_bufs[2]):
            t0 = det_bufs[0][0][0]
            cand1 = min(det_bufs[1], key=lambda x: abs(x[0]-t0))
            cand2 = min(det_bufs[2], key=lambda x: abs(x[0]-t0))
            if abs(cand1[0]-t0)<=max_dt and abs(cand2[0]-t0)<=max_dt:
                _, p0 = det_bufs[0].popleft()
                # pop cand1 and cand2
                for i in range(len(det_bufs[1])):
                    if abs(det_bufs[1][i][0]-cand1[0])<1e-6:
                        _, p1 = det_bufs[1].pop(i); break
                for i in range(len(det_bufs[2])):
                    if abs(det_bufs[2][i][0]-cand2[0])<1e-6:
                        _, p2 = det_bufs[2].pop(i); break
                pixels = [p0,p1,p2]
                X, rms, svals = triangulate_n_cameras(Ks, dists, Rs, Cs, pixels)
                dist_cam1 = np.linalg.norm(X - Cs[0])
                timestamps.append(t0); distances.append(dist_cam1); residuals.append(rms)
                print(f"[{t0:.3f}] dist={dist_cam1:.3f}m rms={rms:.4f}")
        frame_idx += 1

    # write CSV
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s","distance_m","rms_m"])
        for t,d,r in zip(timestamps, distances, residuals):
            w.writerow([t,d,r])

    # Plot
    if timestamps:
        plt.figure()
        plt.plot(timestamps, distances, marker='o')
        plt.xlabel("time (s)"); plt.ylabel("distance (m)"); plt.title("Triangulated distance vs time")
        plt.grid(True); plt.show()
    else:
        print("No matches found / no output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs=3, required=True, help="three video files or camera indexes")
    parser.add_argument("--calibs", nargs=3, required=True, help="three calibration npz files")
    parser.add_argument("--out", default="results.csv", help="output CSV path")
    parser.add_argument("--dets", nargs=3, help="optional comma-separated detection CSVs instead of running detector")
    args = parser.parse_args()
    dets = None
    if args.dets:
        dets = args.dets
    run_adapter(args.cams, args.calibs, args.out, detection_csvs=dets)
