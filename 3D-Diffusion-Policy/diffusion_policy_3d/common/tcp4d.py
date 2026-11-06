"""Shared raw-depth geometry for TCP4D training and deployment."""
import numpy as np
from numba import njit

SPEC = dict(version=1, depth_source='raw', depth_unit='mm', image_shape=[240, 320],
            depth_range=[0.15, 3.0], crop_min=[0., -.7, .45], crop_max=[1.2, .7, 1.4],
            num_points=1024, candidates=4096, seed=42, coordinate_frame='footprint',
            extrinsic_convention='base_to_camera', arm_order=['left', 'right'],
            joint_unit='rad', gripper_unit='total_width_m', action_step_offsets=list(range(8)))

@njit(cache=True, nogil=True)
def fps(points, count):
    selected = np.empty(count, np.int64)
    distances = np.full(len(points), np.inf)
    farthest = 0
    for i in range(count):
        selected[i] = farthest
        for j in range(len(points)):
            distance = 0.
            for k in range(3):
                distance += (points[j,k] - points[farthest,k]) ** 2
            distances[j] = min(distances[j], distance)
        farthest = np.argmax(distances)
    return points[selected]

def point_cloud(depth, intrinsic, extrinsic, spec=SPEC):
    depth = np.asarray(depth)
    if depth.dtype != np.uint16 or list(depth.shape) != spec['image_shape']:
        raise ValueError('depth must be uint16 [240,320], millimeters')
    k, e = np.asarray(intrinsic, dtype=np.float64), np.asarray(extrinsic, dtype=np.float64)
    if k.shape != (3,3) or e.shape != (4,4) or not np.isfinite(k).all() or not np.isfinite(e).all():
        raise ValueError('invalid calibration shapes or values')
    if k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0,0,1]):
        raise ValueError('invalid intrinsic matrix')
    if (not np.allclose(e[3], [0,0,0,1])
            or not np.allclose(e[:3,:3].T @ e[:3,:3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(e[:3,:3]), 1, atol=1e-4)):
        raise ValueError('extrinsic must be a rigid base_to_camera transform')
    inv = np.linalg.inv(e)
    z = depth.astype(np.float64) / 1000
    v,u = np.nonzero((z >= spec['depth_range'][0]) & (z <= spec['depth_range'][1]) & (depth > 0))
    rays = np.column_stack([u,v,np.ones(len(u))]) @ np.linalg.inv(k).T
    cam = rays * z[v,u,None]
    points = cam @ inv[:3,:3].T + inv[:3,3]
    points = points[np.all((points >= spec['crop_min']) & (points <= spec['crop_max']), axis=1)].astype(np.float32)
    if len(points) == 0:
        raise ValueError('no valid raw-depth points inside workspace')
    rng = np.random.default_rng(spec['seed'])
    if len(points) > spec['candidates']:
        points = points[rng.choice(len(points), spec['candidates'], replace=False)]
    if len(points) < spec['num_points']:
        return points[np.arange(spec['num_points']) % len(points)]
    return fps(points, spec['num_points'])
