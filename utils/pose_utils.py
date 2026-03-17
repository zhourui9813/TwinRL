import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    return rot.as_matrix()


def pose6d_to_matrix(vec6: np.ndarray) -> np.ndarray:
    """Convert 6D [tx, ty, tz, r, p, y] to 4x4 homogeneous transform."""

    if vec6.size < 6:
        padded = np.zeros(6, dtype=float)
        padded[: vec6.size] = vec6
        vec6 = padded

    tx, ty, tz, roll, pitch, yaw = vec6[:6]
    T = np.eye(4, dtype=float)
    T[:3, :3] = euler_to_matrix(roll, pitch, yaw)
    T[:3, 3] = [tx, ty, tz]
    return T


def integrate_relative_poses(deltas: np.ndarray, reset_pose: np.ndarray) -> np.ndarray:
    """Integrate relative XYZ increments into world-frame XYZ positions using linear cumulative sum.

    deltas: (T, D>=3) where first 3 are translation (assumed in global frame).
    Rotational increments (if any) are ignored for positioning.
    reset_pose: vector providing initial transform; length 3+.
    Returns: (T, 3) absolute positions.
    """
    if deltas.ndim != 2 or deltas.shape[1] < 3:
        raise ValueError(
            f"Relative pose array must have shape (T, D>=3), got {deltas.shape}"
        )

    reset_pose = np.asarray(reset_pose, dtype=float).flatten()
    start_xyz = np.zeros(3, dtype=float)
    start_xyz[: min(3, reset_pose.size)] = reset_pose[: min(3, reset_pose.size)]

    delta_xyz = deltas[:, :3].astype(float)

    traj_xyz = np.cumsum(delta_xyz, axis=0) + start_xyz

    return traj_xyz