# tools/triangulation.py
"""
N-view triangulation utilities for calibrated cameras.
Works with 3 (or more) cameras. Exposes:
 - pixel_to_bearing(...)
 - triangulate_n_cameras(...)
"""

import numpy as np
import cv2

def undistort_pixel_to_bearing(K, dist, px):
    """Return unit bearing vector in camera coords from pixel (x,y)."""
    px_arr = np.array(px, dtype=np.float32).reshape(1,1,2)
    und = cv2.undistortPoints(px_arr, K, dist, P=None)  # returns normalized coords (x/z, y/z)
    x_n = float(und[0,0,0])
    y_n = float(und[0,0,1])
    v = np.array([x_n, y_n, 1.0], dtype=float)
    v /= np.linalg.norm(v)
    return v

def pixel_to_world_bearing(K, dist, R_cam2world, C_world, px):
    """
    Convert pixel -> unit bearing in WORLD coords.
    R_cam2world: 3x3 that maps camera coords to world coords (u_world = R_cam2world @ u_cam).
    C_world: camera center in world coords (3,)
    """
    u_cam = undistort_pixel_to_bearing(K, dist, px)
    u_world = R_cam2world.dot(u_cam)
    u_world = u_world / np.linalg.norm(u_world)
    return u_world, np.array(C_world).reshape(3,)

def triangulate_n_cameras(Ks, dists, Rs, Cs, pixels):
    """
    Triangulate 3D point from N cameras.
    Ks, dists, Rs, Cs: lists of length N
    pixels: list of (x,y) for each camera (same order)
    Returns:
      X: estimated 3D point (world frame)
      rms: RMS residual distance to rays (meters)
      s_vals: per-camera scalar along each ray
    """
    N = len(Ks)
    assert N >= 2, "need at least two cameras"
    rows = 3 * N
    cols = 3 + N
    A = np.zeros((rows, cols))
    b = np.zeros((rows,))
    u_list = []
    for i in range(N):
        u, C = pixel_to_world_bearing(Ks[i], dists[i], Rs[i], Cs[i], pixels[i])
        u_list.append((u, C))
        r0 = 3*i
        A[r0:r0+3, 0:3] = np.eye(3)
        A[r0:r0+3, 3 + i] = -u
        b[r0:r0+3] = C
    sol, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    X = sol[0:3]
    s = sol[3:]
    dists_to_rays = []
    for i, (u, C) in enumerate(u_list):
        Pi = C + s[i]*u
        dists_to_rays.append(np.linalg.norm(X - Pi))
    rms = float(np.sqrt(np.mean(np.array(dists_to_rays)**2)))
    return X, rms, s
