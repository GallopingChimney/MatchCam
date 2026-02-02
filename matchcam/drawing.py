"""
GPU overlay drawing for MatchCam.

Draws vanishing point lines, control point handles, origin marker,
and status information over the camera view.
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

COL_VP1_LINE = (0.95, 0.2, 0.2, 0.9)       # red (X)
COL_VP1_LINE_EXT = (0.95, 0.2, 0.2, 0.3)
COL_VP1_HANDLE = (1.0, 0.3, 0.3, 1.0)
COL_VP1_HANDLE_HOVER = (1.0, 0.6, 0.5, 1.0)

COL_VP2_LINE = (0.2, 0.85, 0.2, 0.9)       # green (Y)
COL_VP2_LINE_EXT = (0.2, 0.85, 0.2, 0.3)
COL_VP2_HANDLE = (0.3, 0.9, 0.3, 1.0)
COL_VP2_HANDLE_HOVER = (0.6, 1.0, 0.6, 1.0)

COL_VP3_LINE = (0.2, 0.5, 0.95, 0.9)       # blue (Z)
COL_VP3_LINE_EXT = (0.2, 0.5, 0.95, 0.3)
COL_VP3_HANDLE = (0.3, 0.6, 1.0, 1.0)
COL_VP3_HANDLE_HOVER = (0.5, 0.8, 1.0, 1.0)

COL_ORIGIN = (1.0, 0.8, 0.0, 1.0)          # yellow/orange (distinct from VP2 green)
COL_ORIGIN_HOVER = (1.0, 0.9, 0.4, 1.0)

COL_REF_LINE = (0.9, 0.9, 0.2, 0.8)        # yellow
COL_REF_HANDLE = (1.0, 1.0, 0.3, 1.0)
COL_REF_HANDLE_HOVER = (1.0, 1.0, 0.7, 1.0)

COL_VP_INDICATOR = (1.0, 1.0, 1.0, 0.6)    # white

COL_LOUPE_RING = (0.3, 0.9, 0.3, 0.4)     # green, 40% opacity
COL_LOUPE_CROSSHAIR = (1.0, 1.0, 1.0, 0.9)  # white, 90% opacity

HANDLE_RADIUS = 7.0
HANDLE_HOVER_RADIUS = 9.0
LOUPE_RADIUS = 40.0
LOUPE_CROSSHAIR_SIZE = 4.0
LINE_WIDTH = 2.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _circle_verts(cx: float, cy: float, radius: float, segments: int = 16) -> list[tuple[float, float]]:
    """Generate vertices for a filled circle as a triangle fan center list."""
    verts = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        verts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return verts


def _circle_tris(segments: int = 16) -> list[tuple[int, int, int]]:
    """Triangle indices for a fan with center at index 0."""
    tris = []
    for i in range(segments):
        tris.append((0, i + 1, (i + 1) % segments + 1))
    return tris


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
    """Compute camera border rectangle in region pixel coords.

    Returns (x, y, width, height) or None.
    """
    cam = scene.camera
    if cam is None:
        return None

    frame_px = _compute_camera_border_from_projection(
        scene, region, rv3d, cam
    )
    return frame_px


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

    # Get hover state from the running modal operator
    hover_idx = scene.get("_matchcam_hover_idx", -1)

    is_3vp = (props.mode == '3VP')

    # --- Read framebuffer for loupe BEFORE drawing any overlay ---
    loupe_texture = None
    loupe_pos = None
    if scene.get("_matchcam_precision", False):
        drag_screen = scene.get("_matchcam_drag_screen")
        if drag_screen is not None:
            loupe_pos = (drag_screen[0], drag_screen[1])
            loupe_texture = _read_loupe_pixels(loupe_pos[0], loupe_pos[1])

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(LINE_WIDTH)

    # Collect control point screen positions for handle drawing
    def _pt(name):
        v = getattr(props, name)
        return normalized_to_screen(v[0], v[1], frame)

    # --- Draw VP1 lines ---
    _draw_line_pair(
        shader,
        _pt('vp1_line1_start'), _pt('vp1_line1_end'),
        _pt('vp1_line2_start'), _pt('vp1_line2_end'),
        COL_VP1_LINE, COL_VP1_LINE_EXT,
    )

    # --- Draw VP2 lines ---
    _draw_line_pair(
        shader,
        _pt('vp2_line1_start'), _pt('vp2_line1_end'),
        _pt('vp2_line2_start'), _pt('vp2_line2_end'),
        COL_VP2_LINE, COL_VP2_LINE_EXT,
    )

    # --- Draw VP3 lines (3VP mode only) ---
    if is_3vp:
        _draw_line_pair(
            shader,
            _pt('vp3_line1_start'), _pt('vp3_line1_end'),
            _pt('vp3_line2_start'), _pt('vp3_line2_end'),
            COL_VP3_LINE, COL_VP3_LINE_EXT,
        )

    # --- Draw reference distance line ---
    if props.ref_distance_enabled:
        _draw_line(shader, _pt('ref_point_a'), _pt('ref_point_b'), COL_REF_LINE)

    # --- Draw handles ---
    vp1_names = ['vp1_line1_start', 'vp1_line1_end', 'vp1_line2_start', 'vp1_line2_end']
    vp2_names = ['vp2_line1_start', 'vp2_line1_end', 'vp2_line2_start', 'vp2_line2_end']
    vp3_names = ['vp3_line1_start', 'vp3_line1_end', 'vp3_line2_start', 'vp3_line2_end']

    from .properties import CONTROL_POINT_NAMES

    for i, name in enumerate(CONTROL_POINT_NAMES):
        # Skip VP3 handles in 2VP mode
        if name in vp3_names and not is_3vp:
            continue
        if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
            continue

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
        else:
            col = COL_REF_HANDLE_HOVER if is_hover else COL_REF_HANDLE

        radius = HANDLE_HOVER_RADIUS if is_hover else HANDLE_RADIUS
        _draw_filled_circle(shader, pos[0], pos[1], radius, col)

    # --- Draw VP indicators ---
    _draw_vp_indicators(shader, props, frame, is_3vp)

    # --- Draw precision loupe (on top of everything, using pre-read pixels) ---
    if loupe_texture is not None and loupe_pos is not None:
        _draw_loupe(shader, loupe_pos[0], loupe_pos[1], loupe_texture)

    # Restore state
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


def _draw_line(shader, p1, p2, color):
    """Draw a single line segment."""
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": [p1, p2]})
    batch.draw(shader)


def _draw_line_pair(shader, s1, e1, s2, e2, color, ext_color):
    """Draw two line segments with extensions showing convergence."""
    # Main lines
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": [s1, e1, s2, e2]})
    batch.draw(shader)

    # Extensions beyond endpoints
    def _extend(a, b, factor=1.5):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (b[0] + dx * factor, b[1] + dy * factor)

    ext1_far = _extend(s1, e1)
    ext1_near = _extend(e1, s1)
    ext2_far = _extend(s2, e2)
    ext2_near = _extend(e2, s2)

    shader.uniform_float("color", ext_color)
    batch = batch_for_shader(
        shader, 'LINES',
        {"pos": [e1, ext1_far, s1, ext1_near, e2, ext2_far, s2, ext2_near]},
    )
    batch.draw(shader)


def _draw_filled_circle(shader, cx, cy, radius, color, segments=16):
    """Draw a filled circle at screen position."""
    verts = [(cx, cy)]  # center
    verts.extend(_circle_verts(cx, cy, radius, segments))

    indices = _circle_tris(segments)

    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=indices)
    batch.draw(shader)


def _read_loupe_pixels(cx, cy):
    """Read a small region of the framebuffer for the magnified loupe view.

    Returns a gpu.types.GPUTexture or None on failure.
    """
    MAGNIFICATION = 4.0
    sample_size = int(LOUPE_RADIUS * 2 / MAGNIFICATION)
    half_sample = sample_size // 2

    region = bpy.context.region
    x0 = int(cx) - half_sample
    y0 = int(cy) - half_sample

    # Clamp to region bounds
    x0 = max(0, min(x0, region.width - sample_size))
    y0 = max(0, min(y0, region.height - sample_size))

    if sample_size < 2:
        return None

    try:
        fb = gpu.state.active_framebuffer_get()
        buf = fb.read_color(x0, y0, sample_size, sample_size, 4, 0, 'FLOAT')
        buf.dimensions = sample_size * sample_size * 4
        texture = gpu.types.GPUTexture((sample_size, sample_size), data=buf)
        return texture
    except Exception:
        return None


def _draw_loupe(shader, cx, cy, texture):
    """Draw a precision loupe with magnified framebuffer content."""
    segments = 32
    display_half = LOUPE_RADIUS

    # --- Draw magnified texture as a quad ---
    img_shader = gpu.shader.from_builtin('IMAGE')
    img_shader.bind()
    img_shader.uniform_sampler("image", texture)

    batch = batch_for_shader(
        img_shader, 'TRI_FAN',
        {
            "pos": [
                (cx - display_half, cy - display_half),
                (cx + display_half, cy - display_half),
                (cx + display_half, cy + display_half),
                (cx - display_half, cy + display_half),
            ],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        },
    )
    batch.draw(img_shader)

    # --- Re-bind the uniform color shader for ring + crosshairs ---
    shader.bind()

    # Ring outline
    gpu.state.line_width_set(1.0)
    ring_verts = _circle_verts(cx, cy, LOUPE_RADIUS, segments)
    line_pairs = []
    for i in range(segments):
        line_pairs.append(ring_verts[i])
        line_pairs.append(ring_verts[(i + 1) % segments])

    shader.uniform_float("color", COL_LOUPE_RING)
    batch = batch_for_shader(shader, 'LINES', {"pos": line_pairs})
    batch.draw(shader)

    # Crosshairs
    cs = LOUPE_CROSSHAIR_SIZE
    crosshair_verts = [
        (cx - cs, cy), (cx + cs, cy),
        (cx, cy - cs), (cx, cy + cs),
    ]
    shader.uniform_float("color", COL_LOUPE_CROSSHAIR)
    batch = batch_for_shader(shader, 'LINES', {"pos": crosshair_verts})
    batch.draw(shader)

    # Center dot
    _draw_filled_circle(shader, cx, cy, 1.0, COL_LOUPE_CROSSHAIR, segments=8)

    gpu.state.line_width_set(LINE_WIDTH)


def _draw_vp_indicators(shader, props, frame, is_3vp):
    """Draw small diamonds at the computed vanishing point positions."""
    cam = bpy.context.scene.camera
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

    # Compute VPs
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

    vp_list = [(fu, COL_VP1_LINE), (fv, COL_VP2_LINE)]

    if is_3vp:
        fw = _vp_from_lines('vp3_line1_start', 'vp3_line1_end', 'vp3_line2_start', 'vp3_line2_end')
        vp_list.append((fw, COL_VP3_LINE))

    for vp, col in vp_list:
        if vp is None:
            continue

        rel = solv.image_plane_to_relative(vp[0], vp[1], aspect)
        sx, sy = normalized_to_screen(rel[0], rel[1], frame)

        region = bpy.context.region
        if -500 < sx < region.width + 500 and -500 < sy < region.height + 500:
            sz = 6.0
            verts = [
                (sx, sy + sz),
                (sx + sz, sy),
                (sx, sy - sz),
                (sx - sz, sy),
            ]
            indices = [(0, 1, 2), (0, 2, 3)]
            shader.uniform_float("color", (*col[:3], 0.7))
            batch = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=indices)
            batch.draw(shader)


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
