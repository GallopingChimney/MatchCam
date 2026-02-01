"""
Camera calibration solver for MatchCam.

Computes camera focal length, rotation, and translation from two vanishing
points defined by user-placed line segments on an image.

Based on the method described in:
"Using Vanishing Points for Camera Calibration and Coarse 3D Reconstruction
from a Single Image" - Guillou, Meneveaux, Maisel, Bouatouch

Math closely follows fSpy's solver implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def relative_to_image_plane(x: float, y: float, aspect: float) -> tuple[float, float]:
    """Convert normalized (0-1) image coords to image-plane coords.

    Image-plane coords have origin at center, y pointing up,
    and the longer dimension spanning [-1, 1].
    """
    if aspect >= 1.0:  # landscape or square
        return (-1.0 + 2.0 * x, (1.0 - 2.0 * y) / aspect)
    else:  # portrait
        return ((-1.0 + 2.0 * x) * aspect, 1.0 - 2.0 * y)


def image_plane_to_relative(px: float, py: float, aspect: float) -> tuple[float, float]:
    """Convert image-plane coords back to normalized (0-1)."""
    if aspect >= 1.0:
        return ((px + 1.0) / 2.0, (1.0 - py * aspect) / 2.0)
    else:
        return ((px / aspect + 1.0) / 2.0, (1.0 - py) / 2.0)


# ---------------------------------------------------------------------------
# 2D geometry
# ---------------------------------------------------------------------------

def line_intersection(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Compute intersection of lines (p1-p2) and (p3-p4).

    Returns None if lines are parallel (or nearly so).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None

    t1 = x1 * y2 - y1 * x2
    t2 = x3 * y4 - y3 * x4

    ix = (t1 * (x3 - x4) - (x1 - x2) * t2) / denom
    iy = (t1 * (y3 - y4) - (y1 - y2) * t2) / denom
    return (ix, iy)


def vec_sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def vec_add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def vec_scale(v: tuple[float, float], s: float) -> tuple[float, float]:
    return (v[0] * s, v[1] * s)


def vec_dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def vec_length(v: tuple[float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


# ---------------------------------------------------------------------------
# 3D vector helpers
# ---------------------------------------------------------------------------

def vec3_length(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def vec3_normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    ln = vec3_length(v)
    if ln < 1e-15:
        return (0.0, 0.0, 0.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def vec3_cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec3_scale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_add(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# ---------------------------------------------------------------------------
# Matrix helpers (3x3 stored as row-major: m[row][col])
# ---------------------------------------------------------------------------

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def mat3_from_columns(
    c0: tuple[float, float, float],
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
) -> Matrix3:
    return (
        (c0[0], c1[0], c2[0]),
        (c0[1], c1[1], c2[1]),
        (c0[2], c1[2], c2[2]),
    )


def mat3_from_rows(
    r0: tuple[float, float, float],
    r1: tuple[float, float, float],
    r2: tuple[float, float, float],
) -> Matrix3:
    return (r0, r1, r2)


def mat3_determinant(m: Matrix3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def mat3_mul(a: Matrix3, b: Matrix3) -> Matrix3:
    """Standard matrix multiplication: result = a @ b."""
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += a[i][k] * b[k][j]
            row.append(s)
        result.append(tuple(row))
    return tuple(result)


def mat3_vec_mul(m: Matrix3, v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Matrix-vector multiplication: result = m @ v."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mat3_transpose(m: Matrix3) -> Matrix3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


# ---------------------------------------------------------------------------
# Quaternion from rotation matrix
# ---------------------------------------------------------------------------

def quaternion_from_matrix(m: Matrix3) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = m[0][0] + m[1][1] + m[2][2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s

    return (w, x, y, z)


# ---------------------------------------------------------------------------
# Solver result
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    focal_length_mm: float
    rotation_quaternion: tuple[float, float, float, float]  # (w, x, y, z)
    location: tuple[float, float, float]
    hfov_deg: float
    vfov_deg: float
    vp1_image_plane: tuple[float, float]
    vp2_image_plane: tuple[float, float]
    relative_focal_length: float


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_DISTANCE = 10.0


def solve_2vp(
    # VP1 line segments (normalized 0-1 coords)
    vp1_l1_start: tuple[float, float],
    vp1_l1_end: tuple[float, float],
    vp1_l2_start: tuple[float, float],
    vp1_l2_end: tuple[float, float],
    # VP2 line segments
    vp2_l1_start: tuple[float, float],
    vp2_l1_end: tuple[float, float],
    vp2_l2_start: tuple[float, float],
    vp2_l2_end: tuple[float, float],
    # Origin point
    origin: tuple[float, float],
    # Settings
    vp1_axis: str,  # 'X', 'Y', or 'Z'
    vp2_axis: str,
    image_aspect: float,  # width / height
    sensor_width: float = 36.0,  # mm
    # Principal point (normalized 0-1)
    principal_point: tuple[float, float] = (0.5, 0.5),
    # Reference distance
    ref_distance_enabled: bool = False,
    ref_distance: float = 1.0,
    ref_point_a: tuple[float, float] = (0.4, 0.5),
    ref_point_b: tuple[float, float] = (0.6, 0.5),
) -> SolverResult | None:
    """Solve for camera parameters from two vanishing points.

    Returns SolverResult or None if the configuration is invalid.
    """
    # Convert all points to image-plane coordinates
    vp1_s1 = relative_to_image_plane(*vp1_l1_start, image_aspect)
    vp1_e1 = relative_to_image_plane(*vp1_l1_end, image_aspect)
    vp1_s2 = relative_to_image_plane(*vp1_l2_start, image_aspect)
    vp1_e2 = relative_to_image_plane(*vp1_l2_end, image_aspect)

    vp2_s1 = relative_to_image_plane(*vp2_l1_start, image_aspect)
    vp2_e1 = relative_to_image_plane(*vp2_l1_end, image_aspect)
    vp2_s2 = relative_to_image_plane(*vp2_l2_start, image_aspect)
    vp2_e2 = relative_to_image_plane(*vp2_l2_end, image_aspect)

    pp = relative_to_image_plane(*principal_point, image_aspect)
    orig_ip = relative_to_image_plane(*origin, image_aspect)

    # --- Compute vanishing points ---
    fu = line_intersection(vp1_s1, vp1_e1, vp1_s2, vp1_e2)
    fv = line_intersection(vp2_s1, vp2_e1, vp2_s2, vp2_e2)

    if fu is None or fv is None:
        return None  # parallel lines

    # --- Compute focal length (orthogonality constraint) ---
    # Project principal point onto line Fu-Fv
    dir_fuv = vec_sub(fu, fv)
    dir_fuv_len = vec_length(dir_fuv)
    if dir_fuv_len < 1e-12:
        return None  # VPs coincide

    dir_fuv_n = vec_scale(dir_fuv, 1.0 / dir_fuv_len)

    # Projection of P onto line Fu-Fv: Puv
    pp_to_fv = vec_sub(pp, fv)
    proj = vec_dot(dir_fuv_n, pp_to_fv)
    puv = vec_add(fv, vec_scale(dir_fuv_n, proj))

    pp_puv_dist_sq = vec_dot(vec_sub(pp, puv), vec_sub(pp, puv))

    # Signed distances from Puv to each VP along the Fu-Fv direction
    fv_puv_signed = proj
    fu_puv_signed = proj - dir_fuv_len

    f_squared = -(fv_puv_signed * fu_puv_signed) - pp_puv_dist_sq

    if f_squared <= 0:
        return None  # invalid VP configuration

    f = math.sqrt(f_squared)

    # --- Compute FOV ---
    if image_aspect >= 1.0:
        hfov = 2.0 * math.atan(1.0 / f)
        vfov = 2.0 * math.atan(1.0 / (f * image_aspect))
    else:
        hfov = 2.0 * math.atan(image_aspect / f)
        vfov = 2.0 * math.atan(1.0 / f)

    # --- Compute camera rotation matrix ---
    # Vectors from principal point through VPs in camera space (camera looks down -Z)
    ofu = (fu[0] - pp[0], fu[1] - pp[1], -f)
    ofv = (fv[0] - pp[0], fv[1] - pp[1], -f)

    s1 = vec3_length(ofu)
    s2 = vec3_length(ofv)

    if s1 < 1e-12 or s2 < 1e-12:
        return None

    col_u = vec3_normalize(ofu)
    col_v = vec3_normalize(ofv)
    col_w = vec3_cross(col_u, col_v)

    # Camera rotation matrix: columns are the VP directions in camera space
    # This maps from "VP direction space" to camera space
    cam_rot = mat3_from_columns(col_u, col_v, col_w)

    # Ensure proper rotation (det = 1)
    det = mat3_determinant(cam_rot)
    if abs(det) < 1e-6:
        return None

    if det < 0:
        col_w = vec3_scale(col_w, -1.0)
        cam_rot = mat3_from_columns(col_u, col_v, col_w)

    # --- Axis assignment ---
    # Build axis assignment matrix with ROWS as axis vectors (following fSpy)
    # Row 0 = world axis vector for VP1
    # Row 1 = world axis vector for VP2
    # Row 2 = cross product (right-hand rule)
    axis_vectors = {
        'X': (1.0, 0.0, 0.0),
        'Y': (0.0, 1.0, 0.0),
        'Z': (0.0, 0.0, 1.0),
    }

    if vp1_axis == vp2_axis:
        return None

    row0 = axis_vectors[vp1_axis]
    row1 = axis_vectors[vp2_axis]
    row2 = vec3_cross(row0, row1)

    axis_mat = mat3_from_rows(row0, row1, row2)

    # Validate: determinant must be +-1
    axis_det = mat3_determinant(axis_mat)
    if abs(abs(axis_det) - 1.0) > 1e-6:
        return None

    # --- Combine: viewTransform = cam_rot @ axis_mat ---
    # Following fSpy: axisAssignmentMatrix.leftMultiplied(cameraRotationMatrix)
    # which computes cameraRotationMatrix × axisAssignmentMatrix
    # This gives us the world-to-camera rotation matrix
    view_rot = mat3_mul(cam_rot, axis_mat)

    # Ensure det = 1
    view_det = mat3_determinant(view_rot)
    if view_det < 0:
        # Flip third row of axis matrix
        row2 = vec3_scale(row2, -1.0)
        axis_mat = mat3_from_rows(row0, row1, row2)
        view_rot = mat3_mul(cam_rot, axis_mat)

    # --- Compute translation in camera space ---
    # Following fSpy: the origin point is unprojected to camera space
    k = math.tan(0.5 * hfov)

    # Translation = position of world origin in camera space
    translation_cam = (
        k * (orig_ip[0] - pp[0]),
        k * (orig_ip[1] - pp[1]),
        -1.0,
    )
    translation_cam = vec3_scale(translation_cam, DEFAULT_CAMERA_DISTANCE)

    # --- Reference distance scaling ---
    if ref_distance_enabled and ref_distance > 0:
        ra_ip = relative_to_image_plane(*ref_point_a, image_aspect)
        rb_ip = relative_to_image_plane(*ref_point_b, image_aspect)

        # Unproject reference points to camera-space 3D positions
        # at the same depth as the origin
        ref_a_cam = (
            k * (ra_ip[0] - pp[0]),
            k * (ra_ip[1] - pp[1]),
            -1.0,
        )
        ref_a_cam = vec3_scale(ref_a_cam, DEFAULT_CAMERA_DISTANCE)

        ref_b_cam = (
            k * (rb_ip[0] - pp[0]),
            k * (rb_ip[1] - pp[1]),
            -1.0,
        )
        ref_b_cam = vec3_scale(ref_b_cam, DEFAULT_CAMERA_DISTANCE)

        default_dist = vec3_length(vec3_sub(ref_a_cam, ref_b_cam))
        if default_dist > 1e-10:
            scale = ref_distance / default_dist
            translation_cam = vec3_scale(translation_cam, scale)

    # --- Convert to Blender camera parameters ---
    # view_rot is the world-to-camera rotation (columns of view_rot are
    # where world X,Y,Z axes point in camera space).
    #
    # Blender camera convention: -Z forward, +Y up (same as fSpy's camera space).
    # Blender's camera.matrix_world gives the camera-to-world transform.
    #
    # camera-to-world rotation = view_rot^T (inverse = transpose for orthonormal)
    # camera world position = -view_rot^T @ translation_cam

    cam_to_world_rot = mat3_transpose(view_rot)
    location = vec3_scale(mat3_vec_mul(cam_to_world_rot, translation_cam), -1.0)

    # Quaternion from the camera-to-world rotation
    quat = quaternion_from_matrix(cam_to_world_rot)

    # --- Focal length in mm ---
    if image_aspect >= 1.0:
        focal_length_mm = f * sensor_width / 2.0
    else:
        sensor_height = sensor_width / image_aspect
        focal_length_mm = f * sensor_height / 2.0

    return SolverResult(
        focal_length_mm=focal_length_mm,
        rotation_quaternion=quat,
        location=location,
        hfov_deg=math.degrees(hfov),
        vfov_deg=math.degrees(vfov),
        vp1_image_plane=fu,
        vp2_image_plane=fv,
        relative_focal_length=f,
    )
