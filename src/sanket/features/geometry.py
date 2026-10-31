"""Anatomical hand geometry features: joint angles and pairwise distances.

Implements scale- and translation-invariant geometric descriptors for MediaPipe
3D hand skeletons (21 landmarks).
"""

from __future__ import annotations

import numpy as np

# MediaPipe Hand Landmark Indices
WRIST_IDX = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 15 Flexion angle triplets (point_a, joint_vertex, point_b)
FLEXION_ANGLE_TRIPLETS: list[tuple[str, int, int, int]] = [
    # Thumb
    ("thumb_cmc_angle", WRIST_IDX, THUMB_CMC, THUMB_MCP),
    ("thumb_mcp_angle", THUMB_CMC, THUMB_MCP, THUMB_IP),
    ("thumb_ip_angle", THUMB_MCP, THUMB_IP, THUMB_TIP),
    # Index finger
    ("index_mcp_angle", WRIST_IDX, INDEX_MCP, INDEX_PIP),
    ("index_pip_angle", INDEX_MCP, INDEX_PIP, INDEX_DIP),
    ("index_dip_angle", INDEX_PIP, INDEX_DIP, INDEX_TIP),
    # Middle finger
    ("middle_mcp_angle", WRIST_IDX, MIDDLE_MCP, MIDDLE_PIP),
    ("middle_pip_angle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    ("middle_dip_angle", MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    # Ring finger
    ("ring_mcp_angle", WRIST_IDX, RING_MCP, RING_PIP),
    ("ring_pip_angle", RING_MCP, RING_PIP, RING_DIP),
    ("ring_dip_angle", RING_PIP, RING_DIP, RING_TIP),
    # Pinky finger
    ("pinky_mcp_angle", WRIST_IDX, PINKY_MCP, PINKY_PIP),
    ("pinky_pip_angle", PINKY_MCP, PINKY_PIP, PINKY_DIP),
    ("pinky_dip_angle", PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

# 4 Inter-finger abduction angle pairs (vector_1_start, vector_1_end, vector_2_start, vector_2_end)
ABDUCTION_ANGLE_PAIRS: list[tuple[str, int, int, int, int]] = [
    ("thumb_index_abduction", THUMB_CMC, THUMB_TIP, INDEX_MCP, INDEX_TIP),
    ("index_middle_abduction", INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP),
    ("middle_ring_abduction", MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP),
    ("ring_pinky_abduction", RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP),
]

NUM_HAND_ANGLES = len(FLEXION_ANGLE_TRIPLETS) + len(ABDUCTION_ANGLE_PAIRS)  # 19

# 24 Key pairwise Euclidean distance pairs (point_a, point_b)
PAIRWISE_DISTANCE_PAIRS: list[tuple[str, int, int]] = [
    # 5 Fingertips to wrist
    ("dist_thumb_tip_wrist", THUMB_TIP, WRIST_IDX),
    ("dist_index_tip_wrist", INDEX_TIP, WRIST_IDX),
    ("dist_middle_tip_wrist", MIDDLE_TIP, WRIST_IDX),
    ("dist_ring_tip_wrist", RING_TIP, WRIST_IDX),
    ("dist_pinky_tip_wrist", PINKY_TIP, WRIST_IDX),
    # 5 Fingertips to palm center (middle MCP)
    ("dist_thumb_tip_palm", THUMB_TIP, MIDDLE_MCP),
    ("dist_index_tip_palm", INDEX_TIP, MIDDLE_MCP),
    ("dist_middle_tip_palm", MIDDLE_TIP, MIDDLE_MCP),
    ("dist_ring_tip_palm", RING_TIP, MIDDLE_MCP),
    ("dist_pinky_tip_palm", PINKY_TIP, MIDDLE_MCP),
    # 10 All pairwise inter-fingertip distances
    ("dist_thumb_index", THUMB_TIP, INDEX_TIP),
    ("dist_thumb_middle", THUMB_TIP, MIDDLE_TIP),
    ("dist_thumb_ring", THUMB_TIP, RING_TIP),
    ("dist_thumb_pinky", THUMB_TIP, PINKY_TIP),
    ("dist_index_middle", INDEX_TIP, MIDDLE_TIP),
    ("dist_index_ring", INDEX_TIP, RING_TIP),
    ("dist_index_pinky", INDEX_TIP, PINKY_TIP),
    ("dist_middle_ring", MIDDLE_TIP, RING_TIP),
    ("dist_middle_pinky", MIDDLE_TIP, PINKY_TIP),
    ("dist_ring_pinky", RING_TIP, PINKY_TIP),
    # 4 Fingertip to own base MCP (curling metrics, middle_tip to middle_mcp is already above)
    ("dist_thumb_curl", THUMB_TIP, THUMB_MCP),
    ("dist_index_curl", INDEX_TIP, INDEX_MCP),
    ("dist_ring_curl", RING_TIP, RING_MCP),
    ("dist_pinky_curl", PINKY_TIP, PINKY_MCP),
]

NUM_HAND_DISTANCES = len(PAIRWISE_DISTANCE_PAIRS)  # 24


def compute_angle_between_vectors(
    vec1: np.ndarray, vec2: np.ndarray, eps: float = 1e-7
) -> float:
    """Calculate the 3D angle in radians between two vectors.

    Args:
        vec1: Vector of shape (3,).
        vec2: Vector of shape (3,).
        eps: Small epsilon preventing division by zero.

    Returns:
        Angle in radians in range [0, pi].
    """
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 < eps or norm2 < eps:
        return 0.0

    cosine = float(np.dot(vec1, vec2) / (norm1 * norm2))
    cosine_clamped = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine_clamped))


def compute_joint_angles(coords: np.ndarray) -> np.ndarray:
    """Compute 19 anatomical flexion and abduction joint angles for a single hand skeleton.

    Args:
        coords: Hand landmark coordinate array of shape (21, 3).

    Returns:
        1D float32 array of shape (19,) containing angles in radians [0, pi].
        If coords is unobserved (all zeros), returns an array of zeros.
    """
    if coords.ndim != 2 or coords.shape != (21, 3):
        msg = f"Expected shape (21, 3), got {coords.shape}"
        raise ValueError(msg)

    angles = np.zeros(NUM_HAND_ANGLES, dtype=np.float32)

    # If hand is unobserved / completely missing
    if np.all(coords == 0.0):
        return angles

    idx = 0
    # 1. Flexion angles at joint vertices
    for _, pt_a, pt_b, pt_c in FLEXION_ANGLE_TRIPLETS:
        vec_ba = coords[pt_a] - coords[pt_b]
        vec_bc = coords[pt_c] - coords[pt_b]
        angles[idx] = compute_angle_between_vectors(vec_ba, vec_bc)
        idx += 1

    # 2. Inter-finger abduction angles
    for _, start1, end1, start2, end2 in ABDUCTION_ANGLE_PAIRS:
        vec1 = coords[end1] - coords[start1]
        vec2 = coords[end2] - coords[start2]
        angles[idx] = compute_angle_between_vectors(vec1, vec2)
        idx += 1

    return angles


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute 24 key anatomical pairwise Euclidean distances for a single hand.

    Args:
        coords: Hand landmark coordinate array of shape (21, 3).

    Returns:
        1D float32 array of shape (24,) containing Euclidean distances.
        If coords is unobserved (all zeros), returns an array of zeros.
    """
    if coords.ndim != 2 or coords.shape != (21, 3):
        msg = f"Expected shape (21, 3), got {coords.shape}"
        raise ValueError(msg)

    distances = np.zeros(NUM_HAND_DISTANCES, dtype=np.float32)

    if np.all(coords == 0.0):
        return distances

    for idx, (_, pt_a, pt_b) in enumerate(PAIRWISE_DISTANCE_PAIRS):
        diff = coords[pt_a] - coords[pt_b]
        distances[idx] = float(np.linalg.norm(diff))

    return distances


def get_hand_angle_names(prefix: str = "hand") -> list[str]:
    """Return descriptive column names for the 19 hand angle features."""
    names: list[str] = [f"{prefix}_{item[0]}" for item in FLEXION_ANGLE_TRIPLETS]
    names.extend([f"{prefix}_{item[0]}" for item in ABDUCTION_ANGLE_PAIRS])
    return names


def get_hand_distance_names(prefix: str = "hand") -> list[str]:
    """Return descriptive column names for the 24 hand pairwise distance features."""
    return [f"{prefix}_{item[0]}" for item in PAIRWISE_DISTANCE_PAIRS]
