"""
GPU overlay drawing for MatchCam.

Draws vanishing point lines, control point handles, origin marker,
and status information over the camera view.  All primitives use
anti-aliased rendering via the SMOOTH_COLOR shader with feathered edges.
"""

from __future__ import annotations

import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import solver as solv


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_VP1_LINE = (0.95, 0.2, 0.2, 0.8)       # red (X)
COL_VP1_LINE_EXT = (0.95, 0.2, 0.2, 0.15)
COL_VP1_HANDLE = (1.0, 0.3, 0.3, 1.0)
COL_VP1_HANDLE_HOVER = (1.0, 0.6, 0.5, 1.0)

COL_VP2_LINE = (0.2, 0.85, 0.2, 0.8)       # green (Y)
COL_VP2_LINE_EXT = (0.2, 0.85, 0.2, 0.15)
COL_VP2_HANDLE = (0.3, 0.9, 0.3, 1.0)
COL_VP2_HANDLE_HOVER = (0.6, 1.0, 0.6, 1.0)

COL_VP3_LINE = (0.2, 0.5, 0.95, 0.8)       # blue (Z)
COL_VP3_LINE_EXT = (0.2, 0.5, 0.95, 0.15)
COL_VP3_HANDLE = (0.3, 0.6, 1.0, 1.0)
COL_VP3_HANDLE_HOVER = (0.5, 0.8, 1.0, 1.0)

COL_ORIGIN = (1.0, 0.8, 0.0, 1.0)
COL_ORIGIN_HOVER = (1.0, 0.9, 0.4, 1.0)

COL_REF_LINE = (0.9, 0.9, 0.2, 0.8)
COL_REF_HANDLE = (1.0, 1.0, 0.3, 1.0)
COL_REF_HANDLE_HOVER = (1.0, 1.0, 0.7, 1.0)

COL_PP = (1.0, 1.0, 1.0, 0.9)           # white for principal point
COL_PP_HOVER = (1.0, 1.0, 1.0, 1.0)
COL_PP_CROSSHAIR = (1.0, 1.0, 1.0, 0.5)

COL_VP1_FILL = (0.95, 0.2, 0.2, 0.20)
COL_VP2_FILL = (0.2, 0.85, 0.2, 0.20)
COL_VP3_FILL = (0.2, 0.5, 0.95, 0.20)

HANDLE_RADIUS = 7.0
HANDLE_HOVER_RADIUS = 9.0
LOUPE_RADIUS = 48.0
LOUPE_CROSSHAIR_SIZE = 4.0
LOUPE_BORDER_WIDTH = 2.0
LOUPE_MAGNIFICATION = 4.0
LOUPE_SEGMENTS = 64
LINE_WIDTH = 1.5
AA_FRINGE = 1.0  # anti-aliasing feather width in pixels


# ---------------------------------------------------------------------------
# Anti-aliased drawing primitives
# ---------------------------------------------------------------------------

def _draw_aa_line(p1, p2, color, width=LINE_WIDTH, color_end=None):
    """Draw an anti-aliased line as a quad with feathered edges.

    If *color_end* is given the line linearly interpolates from *color*
    at p1 to *color_end* at p2.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return

    nx = -dy / length
    ny = dx / length

    hw = width / 2.0
    aa = AA_FRINGE

    c1 = tuple(color)
    c2 = tuple(color_end) if color_end is not None else c1
    c1_outer = (c1[0], c1[1], c1[2], 0.0)
    c2_outer = (c2[0], c2[1], c2[2], 0.0)

    verts = [
        (p1[0] + nx * (hw + aa), p1[1] + ny * (hw + aa)),
        (p1[0] + nx * hw, p1[1] + ny * hw),
        (p1[0] - nx * hw, p1[1] - ny * hw),
        (p1[0] - nx * (hw + aa), p1[1] - ny * (hw + aa)),
        (p2[0] + nx * (hw + aa), p2[1] + ny * (hw + aa)),
        (p2[0] + nx * hw, p2[1] + ny * hw),
        (p2[0] - nx * hw, p2[1] - ny * hw),
        (p2[0] - nx * (hw + aa), p2[1] - ny * (hw + aa)),
    ]

    colors = [
        c1_outer, c1, c1, c1_outer,
        c2_outer, c2, c2, c2_outer,
    ]

    indices = [
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
    ]

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    shader.bind()
    batch = batch_for_shader(shader, 'TRIS',
        {"pos": verts, "color": colors}, indices=indices)
    batch.draw(shader)


def _draw_aa_circle(cx, cy, radius, color, segments=32):
    """Draw an anti-aliased filled circle with feathered edge."""
    aa = AA_FRINGE
    inner_color = tuple(color)
    outer_color = (color[0], color[1], color[2], 0.0)

    verts = [(cx, cy)]
    colors_list = [inner_color]

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        verts.append((cx + radius * cos_a, cy + radius * sin_a))
        colors_list.append(inner_color)
        verts.append((cx + (radius + aa) * cos_a, cy + (radius + aa) * sin_a))
        colors_list.append(outer_color)

    indices = []
    for i in range(segments):
        ni = (i + 1) % segments
        inner_i = 1 + i * 2
        inner_ni = 1 + ni * 2
        outer_i = 2 + i * 2
        outer_ni = 2 + ni * 2
        indices.append((0, inner_i, inner_ni))
        indices.append((inner_i, outer_i, outer_ni))
        indices.append((inner_i, outer_ni, inner_ni))

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    shader.bind()
    batch = batch_for_shader(shader, 'TRIS',
        {"pos": verts, "color": colors_list}, indices=indices)
    batch.draw(shader)


def _draw_aa_annulus(cx, cy, inner_r, outer_r, color, segments=64):
    """Draw an anti-aliased ring (annulus) with feathered edges."""
    aa = AA_FRINGE
    inner_color = tuple(color)
    outer_color = (color[0], color[1], color[2], 0.0)

    verts = []
    colors_list = []

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        verts.append((cx + max(0, inner_r - aa) * cos_a, cy + max(0, inner_r - aa) * sin_a))
        colors_list.append(outer_color)
        verts.append((cx + inner_r * cos_a, cy + inner_r * sin_a))
        colors_list.append(inner_color)
        verts.append((cx + outer_r * cos_a, cy + outer_r * sin_a))
        colors_list.append(inner_color)
        verts.append((cx + (outer_r + aa) * cos_a, cy + (outer_r + aa) * sin_a))
        colors_list.append(outer_color)

    indices = []
    for i in range(segments):
        ni = (i + 1) % segments
        base_i = i * 4
        base_ni = ni * 4
        indices.append((base_i, base_i + 1, base_ni + 1))
        indices.append((base_i, base_ni + 1, base_ni))
        indices.append((base_i + 1, base_i + 2, base_ni + 2))
        indices.append((base_i + 1, base_ni + 2, base_ni + 1))
        indices.append((base_i + 2, base_i + 3, base_ni + 3))
        indices.append((base_i + 2, base_ni + 3, base_ni + 2))

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    shader.bind()
    batch = batch_for_shader(shader, 'TRIS',
        {"pos": verts, "color": colors_list}, indices=indices)
    batch.draw(shader)


def _draw_aa_diamond(cx, cy, size, color):
    """Draw an anti-aliased diamond shape."""
    aa = AA_FRINGE
    inner_color = tuple(color)
    outer_color = (color[0], color[1], color[2], 0.0)

    # Inner diamond + outer AA diamond
    verts = [
        (cx, cy),
        (cx, cy + size), (cx + size, cy), (cx, cy - size), (cx - size, cy),
        (cx, cy + size + aa), (cx + size + aa, cy),
        (cx, cy - size - aa), (cx - size - aa, cy),
    ]
    colors_list = [
        inner_color,
        inner_color, inner_color, inner_color, inner_color,
        outer_color, outer_color, outer_color, outer_color,
    ]
    indices = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    shader.bind()
    batch = batch_for_shader(shader, 'TRIS',
        {"pos": verts, "color": colors_list}, indices=indices)
    batch.draw(shader)


# ---------------------------------------------------------------------------
# Camera frame mapping
# ---------------------------------------------------------------------------

def get_camera_frame_px(context) -> tuple[float, float, float, float] | None:
    """Get camera frame in region pixel coordinates: (x, y, width, height).

    Returns None if not in camera view.
    """
    rv3d = context.region_data
    if rv3d is None or not rv3d.view_perspective == 'CAMERA':
        return None

    scene = context.scene
    region = context.region

    frame = _view3d_camera_border(scene, region, rv3d)
    if frame is None:
        return None

    return frame


def _view3d_camera_border(scene, region, rv3d):
    """Compute camera border rectangle in region pixel coords."""
    cam = scene.camera
    if cam is None:
        return None

    return _compute_camera_border_from_projection(scene, region, rv3d, cam)


def _compute_camera_border_from_projection(scene, region, rv3d, camera):
    """Compute camera frame corners by projecting known camera-space points."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    cam_data = camera.data
    cam_matrix = camera.matrix_world

    frame = cam_data.view_frame(scene=scene)

    corners_2d = []
    for corner in frame:
        world_pos = cam_matrix @ corner
        pos_2d = location_3d_to_region_2d(region, rv3d, world_pos)
        if pos_2d is None:
            return None
        corners_2d.append(pos_2d)

    xs = [c.x for c in corners_2d]
    ys = [c.y for c in corners_2d]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def normalized_to_screen(
    nx: float, ny: float, frame: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Convert normalized (0-1) image coords to screen pixels."""
    fx, fy, fw, fh = frame
    return (fx + nx * fw, fy + (1.0 - ny) * fh)


def screen_to_normalized(
    sx: float, sy: float, frame: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Convert screen pixels to normalized (0-1) image coords."""
    fx, fy, fw, fh = frame
    if fw < 1 or fh < 1:
        return (0.5, 0.5)
    return ((sx - fx) / fw, 1.0 - (sy - fy) / fh)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_draw_handler = None


def draw_callback(context):
    """Main draw callback for the MatchCam overlay."""
    scene = context.scene
    if not hasattr(scene, 'matchcam'):
        return

    props = scene.matchcam
    if not props.enabled:
        return

    frame = get_camera_frame_px(context)
    if frame is None:
        return

    hover_idx = scene.get("_matchcam_hover_idx", -1)
    drag_idx = scene.get("_matchcam_drag_idx", -1)
    is_3vp = (props.mode == '3VP')

    gpu.state.blend_set('ALPHA')

    def _pt(name):
        v = getattr(props, name)
        return normalized_to_screen(v[0], v[1], frame)

    # --- Draw VP fills (behind lines) ---
    show_fill = props.show_fill
    vp_drag = scene.get("_matchcam_vp_drag", -1)

    # Determine active VP group from drag index or VP diamond drag
    active_vp = -1
    if 0 <= drag_idx <= 3 or vp_drag == 0:
        active_vp = 0
    elif 4 <= drag_idx <= 7 or vp_drag == 1:
        active_vp = 1
    elif 8 <= drag_idx <= 11 or vp_drag == 2:
        active_vp = 2

    def _fill_color(base, vp_idx):
        alpha = 0.05 + 0.03 if (show_fill and vp_idx == active_vp) else 0.05
        return (base[0], base[1], base[2], min(alpha, 1.0))

    if show_fill:
        _draw_vp_fill(
            _pt('vp1_line1_start'), _pt('vp1_line1_end'),
            _pt('vp1_line2_start'), _pt('vp1_line2_end'),
            _fill_color(COL_VP1_FILL, 0),
        )
        _draw_vp_fill(
            _pt('vp2_line1_start'), _pt('vp2_line1_end'),
            _pt('vp2_line2_start'), _pt('vp2_line2_end'),
            _fill_color(COL_VP2_FILL, 1),
        )
        if is_3vp:
            _draw_vp_fill(
                _pt('vp3_line1_start'), _pt('vp3_line1_end'),
                _pt('vp3_line2_start'), _pt('vp3_line2_end'),
                _fill_color(COL_VP3_FILL, 2),
            )

    # --- Draw VP1 lines ---
    _draw_aa_line_pair(
        _pt('vp1_line1_start'), _pt('vp1_line1_end'),
        _pt('vp1_line2_start'), _pt('vp1_line2_end'),
        COL_VP1_LINE, COL_VP1_LINE_EXT,
    )

    # --- Draw VP2 lines ---
    _draw_aa_line_pair(
        _pt('vp2_line1_start'), _pt('vp2_line1_end'),
        _pt('vp2_line2_start'), _pt('vp2_line2_end'),
        COL_VP2_LINE, COL_VP2_LINE_EXT,
    )

    # --- Draw VP3 lines (3VP mode only) ---
    if is_3vp:
        _draw_aa_line_pair(
            _pt('vp3_line1_start'), _pt('vp3_line1_end'),
            _pt('vp3_line2_start'), _pt('vp3_line2_end'),
            COL_VP3_LINE, COL_VP3_LINE_EXT,
        )

    # --- Draw reference distance line ---
    if props.ref_distance_enabled:
        _draw_aa_line(_pt('ref_point_a'), _pt('ref_point_b'), COL_REF_LINE)

    # --- Draw handles ---
    vp1_names = ['vp1_line1_start', 'vp1_line1_end', 'vp1_line2_start', 'vp1_line2_end']
    vp2_names = ['vp2_line1_start', 'vp2_line1_end', 'vp2_line2_start', 'vp2_line2_end']
    vp3_names = ['vp3_line1_start', 'vp3_line1_end', 'vp3_line2_start', 'vp3_line2_end']

    from .properties import CONTROL_POINT_NAMES

    pp_active = (props.mode == '2VP' and props.use_custom_pp)

    for i, name in enumerate(CONTROL_POINT_NAMES):
        if name in vp3_names and not is_3vp:
            continue
        if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
            continue
        if name == 'principal_point':
            continue  # drawn separately as reticle

        pos = _pt(name)
        is_hover = (i == hover_idx)

        if name in vp1_names:
            col = COL_VP1_HANDLE_HOVER if is_hover else COL_VP1_HANDLE
        elif name in vp2_names:
            col = COL_VP2_HANDLE_HOVER if is_hover else COL_VP2_HANDLE
        elif name in vp3_names:
            col = COL_VP3_HANDLE_HOVER if is_hover else COL_VP3_HANDLE
        elif name == 'origin_point':
            col = COL_ORIGIN_HOVER if is_hover else COL_ORIGIN
            # Origin: 2x radius, stroke ring only (no fill), 2px stroke
            origin_r = HANDLE_HOVER_RADIUS * 2 if is_hover else HANDLE_RADIUS * 2
            _draw_aa_annulus(pos[0], pos[1], origin_r - 1.0, origin_r + 1.0, col, segments=32)
            continue
        else:
            col = COL_REF_HANDLE_HOVER if is_hover else COL_REF_HANDLE

        radius = HANDLE_HOVER_RADIUS if is_hover else HANDLE_RADIUS

        if i == drag_idx:
            # The actively dragged VP handle: ring style with center dot
            ring_r = HANDLE_HOVER_RADIUS * 2
            _draw_aa_annulus(pos[0], pos[1], ring_r - 1.0, ring_r + 1.0, col, segments=32)
            _draw_aa_circle(pos[0], pos[1], 2.0, col, segments=12)
        else:
            _draw_aa_circle(pos[0], pos[1], radius, col)

    # --- Draw VP indicators ---
    _draw_vp_indicators(props, frame, is_3vp)

    # --- Draw principal point indicator (2VP + custom PP only) ---
    if pp_active:
        _draw_principal_point(props, frame, hover_idx)

    # --- Draw precision loupe (on top of everything) ---
    if scene.get("_matchcam_precision", False):
        drag_screen = scene.get("_matchcam_drag_screen")
        drag_idx = scene.get("_matchcam_drag_idx", -1)
        if drag_screen is not None:
            _draw_loupe(drag_screen[0], drag_screen[1], frame, drag_idx)

    gpu.state.blend_set('NONE')


def _line_intersect_2d(s1, e1, s2, e2):
    """Intersect two infinite 2D lines. Returns (x, y) or None if parallel."""
    d1x, d1y = e1[0] - s1[0], e1[1] - s1[1]
    d2x, d2y = e2[0] - s2[0], e2[1] - s2[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-10:
        return None
    t = ((s2[0] - s1[0]) * d2y - (s2[1] - s1[1]) * d2x) / denom
    return (s1[0] + t * d1x, s1[1] + t * d1y)


def _draw_vp_fill(s1, e1, s2, e2, fill_color, ext_factor=0.75):
    """Draw a triangular filled area between two VP line segments.

    The triangle runs from the VP intersection (apex) to the farthest
    endpoints of each line (base).  Extension beyond the base fades
    from *fill_color* to transparent.
    """
    vp = _line_intersect_2d(s1, e1, s2, e2)
    if vp is None:
        return

    # Pick the farthest endpoint from the VP for each line
    d_s1 = (s1[0] - vp[0]) ** 2 + (s1[1] - vp[1]) ** 2
    d_e1 = (e1[0] - vp[0]) ** 2 + (e1[1] - vp[1]) ** 2
    far1 = s1 if d_s1 >= d_e1 else e1

    d_s2 = (s2[0] - vp[0]) ** 2 + (s2[1] - vp[1]) ** 2
    d_e2 = (e2[0] - vp[0]) ** 2 + (e2[1] - vp[1]) ** 2
    far2 = s2 if d_s2 >= d_e2 else e2

    fc = tuple(fill_color)
    ft = (fc[0], fc[1], fc[2], 0.0)

    # Extension tips beyond the far endpoints (away from VP)
    def _extend(origin, tip, factor):
        return (tip[0] + (tip[0] - origin[0]) * factor,
                tip[1] + (tip[1] - origin[1]) * factor)

    ext1 = _extend(vp, far1, ext_factor)
    ext2 = _extend(vp, far2, ext_factor)

    # Vertices:  VP(0)  far1(1)  far2(2)  ext1(3)  ext2(4)
    verts = [vp, far1, far2, ext1, ext2]
    colors = [fc, fc, fc, ft, ft]
    indices = [
        # Core triangle
        (0, 1, 2),
        # Extension quad (two tris) from base to faded tips
        (1, 3, 4), (1, 4, 2),
    ]

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    shader.bind()
    batch = batch_for_shader(shader, 'TRIS',
        {"pos": verts, "color": colors}, indices=indices)
    batch.draw(shader)


def _draw_aa_line_pair(s1, e1, s2, e2, color, ext_color):
    """Draw two line segments with AA extensions that fade to transparent."""
    _draw_aa_line(s1, e1, color)
    _draw_aa_line(s2, e2, color)

    # Extension color fades linearly from ext_color to fully transparent
    ext_transparent = (ext_color[0], ext_color[1], ext_color[2], 0.0)

    def _extend(a, b, factor=1.5):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (b[0] + dx * factor, b[1] + dy * factor)

    _draw_aa_line(e1, _extend(s1, e1), ext_color, color_end=ext_transparent)
    _draw_aa_line(s1, _extend(e1, s1), ext_color, color_end=ext_transparent)
    _draw_aa_line(e2, _extend(s2, e2), ext_color, color_end=ext_transparent)
    _draw_aa_line(s2, _extend(e2, s2), ext_color, color_end=ext_transparent)


# ---------------------------------------------------------------------------
# Loupe
# ---------------------------------------------------------------------------

def _get_loupe_color(drag_idx):
    """Get the VP axis color for the dragged handle index."""
    if 0 <= drag_idx <= 3:
        return COL_VP1_LINE[:3]   # red
    elif 4 <= drag_idx <= 7:
        return COL_VP2_LINE[:3]   # green
    elif 8 <= drag_idx <= 11:
        return COL_VP3_LINE[:3]   # blue
    elif drag_idx == 12:
        return COL_ORIGIN[:3]     # yellow
    elif drag_idx == 15:
        return COL_PP[:3]         # white (principal point)
    else:
        return COL_REF_LINE[:3]   # yellow


def _draw_loupe(cx, cy, frame, drag_idx):
    """Draw a precision loupe sampling directly from the background image.

    The magnified view is clipped to a circular shape, with an axis-colored
    border ring and crosshairs.
    """
    cam = bpy.context.scene.camera
    if cam is None:
        return
    bg_images = cam.data.background_images
    if not bg_images or bg_images[0].image is None:
        return

    img = bg_images[0].image
    img_w, img_h = img.size
    if img_w == 0 or img_h == 0:
        return

    # Get GPU texture from Blender image (efficient, no pixel copy)
    try:
        texture = gpu.texture.from_image(img)
    except Exception:
        return
    if texture is None:
        return

    # --- Compute UV region for magnification ---
    nx, ny = screen_to_normalized(cx, cy, frame)
    # Image UV space: (0,0) = bottom-left, (1,1) = top-right
    # Our normalized: ny=0 is top, ny=1 is bottom
    uv_cx = nx
    uv_cy = 1.0 - ny

    fx, fy, fw, fh = frame
    if fw < 1 or fh < 1:
        return

    uv_half_x = (LOUPE_RADIUS / LOUPE_MAGNIFICATION) / fw
    uv_half_y = (LOUPE_RADIUS / LOUPE_MAGNIFICATION) / fh

    # --- Draw textured circle (circular clip) ---
    segments = LOUPE_SEGMENTS
    positions = [(cx, cy)]
    uvs = [(uv_cx, uv_cy)]

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        positions.append((cx + LOUPE_RADIUS * cos_a, cy + LOUPE_RADIUS * sin_a))
        uvs.append((uv_cx + uv_half_x * cos_a, uv_cy + uv_half_y * sin_a))

    indices = []
    for i in range(segments):
        ni = (i + 1) % segments
        indices.append((0, i + 1, ni + 1))

    img_shader = gpu.shader.from_builtin('IMAGE')
    img_shader.bind()
    img_shader.uniform_sampler("image", texture)

    batch = batch_for_shader(
        img_shader, 'TRIS',
        {"pos": positions, "texCoord": uvs},
        indices=indices,
    )
    batch.draw(img_shader)

    # --- Border ring (2px, axis-colored, AA) ---
    loupe_rgb = _get_loupe_color(drag_idx)
    border_color = (*loupe_rgb, 0.8)
    crosshair_color = (*loupe_rgb, 0.9)

    inner_r = LOUPE_RADIUS - LOUPE_BORDER_WIDTH / 2
    outer_r = LOUPE_RADIUS + LOUPE_BORDER_WIDTH / 2
    _draw_aa_annulus(cx, cy, inner_r, outer_r, border_color, segments)

    # --- Crosshairs (pixel-aligned for symmetry) ---
    rcx = round(cx * 2) / 2
    rcy = round(cy * 2) / 2
    cs = LOUPE_CROSSHAIR_SIZE

    # Horizontal crosshair
    _draw_aa_line((rcx - cs, rcy), (rcx + cs, rcy), crosshair_color, width=1.0)
    # Vertical crosshair
    _draw_aa_line((rcx, rcy - cs), (rcx, rcy + cs), crosshair_color, width=1.0)

    # Center dot
    _draw_aa_circle(rcx, rcy, 1.0, crosshair_color, segments=12)


# ---------------------------------------------------------------------------
# Principal point indicator
# ---------------------------------------------------------------------------

def _draw_principal_point(props, frame, hover_idx):
    """Draw a crosshair reticle at the custom principal point location."""
    from .properties import CONTROL_POINT_NAMES
    pp_idx = CONTROL_POINT_NAMES.index("principal_point")
    is_hover = (hover_idx == pp_idx)

    pp = props.principal_point
    sx, sy = normalized_to_screen(pp[0], pp[1], frame)

    col = COL_PP_HOVER if is_hover else COL_PP
    cross_col = (1.0, 1.0, 1.0, 0.7) if is_hover else COL_PP_CROSSHAIR

    # Crosshair lines (14px arms when hovered, 12px normal)
    arm = 14.0 if is_hover else 12.0
    _draw_aa_line((sx - arm, sy), (sx + arm, sy), cross_col, width=1.0)
    _draw_aa_line((sx, sy - arm), (sx, sy + arm), cross_col, width=1.0)

    # Center circle
    _draw_aa_circle(sx, sy, 3.0, col)

    # Outer ring
    ring_r = 7.0 if is_hover else 6.0
    _draw_aa_annulus(sx, sy, ring_r - 1.0, ring_r, col, segments=32)


# ---------------------------------------------------------------------------
# VP indicators
# ---------------------------------------------------------------------------

def _draw_vp_indicators(props, frame, is_3vp):
    """Draw small diamonds at the computed vanishing point positions.

    Diamonds grow and brighten on hover and switch to ring+dot when dragged.
    """
    scene = bpy.context.scene
    vp_hover = scene.get("_matchcam_vp_hover", -1)
    vp_drag = scene.get("_matchcam_vp_drag", -1)

    cam = scene.camera
    if cam is None:
        return

    bg_images = cam.data.background_images
    if not bg_images:
        return

    bg = bg_images[0]
    if bg.image is None:
        return

    w, h = bg.image.size
    if w == 0 or h == 0:
        return

    aspect = w / h

    def _vp_from_lines(s1_name, e1_name, s2_name, e2_name):
        s1 = getattr(props, s1_name)
        e1 = getattr(props, e1_name)
        s2 = getattr(props, s2_name)
        e2 = getattr(props, e2_name)
        ip_s1 = solv.relative_to_image_plane(s1[0], s1[1], aspect)
        ip_e1 = solv.relative_to_image_plane(e1[0], e1[1], aspect)
        ip_s2 = solv.relative_to_image_plane(s2[0], s2[1], aspect)
        ip_e2 = solv.relative_to_image_plane(e2[0], e2[1], aspect)
        return solv.line_intersection(ip_s1, ip_e1, ip_s2, ip_e2)

    fu = _vp_from_lines('vp1_line1_start', 'vp1_line1_end', 'vp1_line2_start', 'vp1_line2_end')
    fv = _vp_from_lines('vp2_line1_start', 'vp2_line1_end', 'vp2_line2_start', 'vp2_line2_end')

    vp_list = [(0, fu, COL_VP1_LINE), (1, fv, COL_VP2_LINE)]

    if is_3vp:
        fw_vp = _vp_from_lines('vp3_line1_start', 'vp3_line1_end', 'vp3_line2_start', 'vp3_line2_end')
        vp_list.append((2, fw_vp, COL_VP3_LINE))

    for idx, vp, col in vp_list:
        if vp is None:
            continue

        rel = solv.image_plane_to_relative(vp[0], vp[1], aspect)
        sx, sy = normalized_to_screen(rel[0], rel[1], frame)

        region = bpy.context.region
        if not (-500 < sx < region.width + 500 and -500 < sy < region.height + 500):
            continue

        is_hover = (idx == vp_hover)
        is_drag = (idx == vp_drag)

        if is_drag:
            # Ring + center dot when being dragged
            ring_r = HANDLE_HOVER_RADIUS * 2
            _draw_aa_annulus(sx, sy, ring_r - 1.0, ring_r + 1.0, (*col[:3], 1.0), segments=32)
            _draw_aa_circle(sx, sy, 2.0, (*col[:3], 1.0), segments=12)
        elif is_hover:
            _draw_aa_diamond(sx, sy, 8.0, (*col[:3], 1.0))
        else:
            _draw_aa_diamond(sx, sy, 6.0, (*col[:3], 0.7))


# ---------------------------------------------------------------------------
# Handler management
# ---------------------------------------------------------------------------

def register_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        return
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback, (bpy.context,), 'WINDOW', 'POST_PIXEL'
    )


def unregister_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
