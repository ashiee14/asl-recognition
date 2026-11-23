"""
landmarks.py

Shared utility used by every other script.
- create_detector(): loads MediaPipe's pretrained HandLandmarker model.
- extract_landmarks(): converts MediaPipe's output into a feature vector
  a classifier can use.
- normalize_landmarks(): does the heavy lifting -- see docstring below.

You do not need to train anything here — this model is pretrained by Google.
"""

import math
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


def create_detector(model_path="hand_landmarker.task"):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_points(result):
    """Return the raw list of 21 (x, y, z) tuples for the first detected
    hand, or None if no hand was found. Shared by extract_landmarks,
    extract_landmarks_raw, and gesture tracking (for Z/J)."""
    if not result.hand_landmarks:
        return None
    pts = result.hand_landmarks[0]
    return [(p.x, p.y, p.z) for p in pts]


def extract_landmarks(result):
    """Return a normalized feature vector for the first detected hand,
    or None if no hand was found in the frame. Used for LIVE prediction."""
    raw = extract_hand_points(result)
    if raw is None:
        return None
    return normalize_landmarks(raw)


def extract_landmarks_raw(result):
    """Return the raw, un-normalized 63-value (21 points x,y,z) vector,
    or None if no hand was found. Used only when COLLECTING training data
    -- keeping data.csv in raw form means we can always regenerate the
    processed/normalized version later without losing anything, even if
    the feature engineering (normalize_landmarks) changes again."""
    raw = extract_hand_points(result)
    if raw is None:
        return None
    return [c for p in raw for c in p]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5


def _unit(a):
    n = _norm(a)
    if n < 1e-9:
        n = 1e-9
    return (a[0] / n, a[1] / n, a[2] / n)


def _angle_deg(a, b, c):
    """Angle at point b, between vectors b->a and b->c, in degrees.
    Straight finger ~= 180 degrees, curled finger = much smaller."""
    v1 = _sub(a, b)
    v2 = _sub(c, b)
    n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos_ang = _dot(v1, v2) / (n1 * n2)
    cos_ang = max(-1.0, min(1.0, cos_ang))
    return math.degrees(math.acos(cos_ang))


# (mcp, pip, tip) landmark indices per finger, used to measure how curled
# each finger is
_FINGER_JOINTS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}





def normalize_landmarks(points):
    """points: list of 21 (x, y, z) tuples.

    Does three things so the classifier sees pure handshape:
      1. TRANSLATION: re-centers on the wrist (landmark 0).
      2. ROTATION: builds a local coordinate frame from the wrist and two
         knuckles (index MCP, pinky MCP) so hand tilt/rotation in front of
         the camera doesn't get mistaken for a shape difference. This is
         what stabilizes letters that "wobble" as your wrist moves.
      3. SCALE: divides by hand size (wrist-to-middle-knuckle distance) so
         distance from the camera doesn't matter.

    Also appends two kinds of engineered features:
      - 4 thumb-to-fingertip distances: how close the thumb is tucked to
        each other finger. Separates handshapes like M / N / S, and D
        from a closed fist.
      - 5 finger curl angles: how bent each finger is (straight ~180
        degrees, curled = small angle). Separates a uniformly-curved
        handshape like C from a mostly-curled-but-one-finger-straight
        handshape like D, which distance features alone don't capture
        well.

    Returns a flat list of 72 values (63 coordinates + 4 distances + 5
    curl angles).
    """
    wrist = points[0]
    centered = [_sub(p, wrist) for p in points]

    index_mcp = centered[5]
    pinky_mcp = centered[17]
    mid_mcp = centered[9]

    x_axis = _unit(index_mcp)
    z_axis = _unit(_cross(index_mcp, pinky_mcp))
    y_axis = _unit(_cross(z_axis, x_axis))

    scale = _norm(mid_mcp)
    if scale < 1e-6:
        scale = 1e-6

    local = [
        (_dot(p, x_axis) / scale, _dot(p, y_axis) / scale, _dot(p, z_axis) / scale)
        for p in centered
    ]

    thumb_tip = local[4]
    fingertip_distances = [
        _norm(_sub(thumb_tip, local[tip])) for tip in (8, 12, 16, 20)
    ]

    curl_angles = [
        _angle_deg(local[mcp], local[pip], local[tip])
        for mcp, pip, tip in _FINGER_JOINTS.values()
    ]

    flat = [c for p in local for c in p]
    return flat + fingertip_distances + curl_angles