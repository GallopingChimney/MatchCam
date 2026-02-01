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

COL_VP1_LINE = (0.95, 0.3, 0.2, 0.9)       # red
COL_VP1_LINE_EXT = (0.95, 0.3, 0.2, 0.3)   # red faded
COL_VP1_HANDLE = (1.0, 0.4, 0.3, 1.0)
COL_VP1_HANDLE_HOVER = (1.0, 0.7, 0.5, 1.0)

COL_VP2_LINE = (0.2, 0.5, 0.95, 0.9)       # blue
COL_VP2_LINE_EXT = (0.2, 0.5, 0.95, 0.3)   # blue faded
COL_VP2_HANDLE = (0.3, 0.6, 1.0, 1.0)
COL_VP2_HANDLE_HOVER = (0.5, 0.8, 1.0, 1.0)

COL_ORIGIN = (0.2, 0.9, 0.3, 1.0)          # green
COL_ORIGIN_HOVER = (0.5, 1.0, 0.6, 1.0)

COL_REF_LINE = (0.9, 0.9, 0.2, 0.8)        # yellow
COL_REF_HANDLE = (1.0, 1.0, 0.3, 1.0)
COL_REF_HANDLE_HOVER = (1.0, 1.0, 0.7, 1.0)

COL_VP_INDICATOR = (1.0, 1.0, 1.0, 0.6)    # white

HANDLE_RADIUS = 7.0
HANDLE_HOVER_RADIUS = 9.0
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

    # Get camera border in region space
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

    # Use the view3d utility to get camera frame
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from mathutils import Vector

    # Get render resolution
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    aspect = render.pixel_aspect_x / render.pixel_aspect_y

    # Camera frame corners in normalized device coordinates
    # We compute the frame by projecting the camera's view corners
    cam_data = cam.data

    # Use the view mapping approach
    # In camera view, Blender maps the camera sensor to a specific screen region
    # We can find it by checking view2d or using the frame coordinates

    # Simpler approach: use the region dimensions and view mapping
    # The camera frame in region space can be found via the projection matrix
    import bpy

    # Get camera frame using Blender's own function
    frame_px = _compute_camera_border_from_projection(
        scene, region, rv3d, cam
    )
    return frame_px


def _compute_camera_border_from_projection(scene, region, rv3d, camera):
    """Compute camera frame corners by projecting known camera-space points."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from mathutils import Vector

    # Project camera's own position - this gives us the center
    # Better: project 4 corners of the camera frame at distance 1
    cam_data = camera.data
    cam_matrix = camera.matrix_world

    # Get the camera frame corners at distance 1
    frame = cam_data.view_frame(scene=scene)
    # frame returns 4 Vector corners in camera local space

    corners_2d = []
    for corner in frame:
        world_pos = cam_matrix @ corner
        pos_2d = location_3d_to_region_2d(region, rv3d, world_pos)
        if pos_2d is None:
            return None
        corners_2d.append(pos_2d)

    # Find bounding box
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
    return (fx + nx * fw, fy + (1.0 - ny) * fh)  # flip Y: 0=top in normalized, bottom in screen


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
    hover_idx = getattr(scene, '_matchcam_hover_idx', -1)

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

    # --- Draw reference distance line ---
    if props.ref_distance_enabled:
        _draw_line(shader, _pt('ref_point_a'), _pt('ref_point_b'), COL_REF_LINE)

    # --- Draw handles ---
    vp1_names = ['vp1_line1_start', 'vp1_line1_end', 'vp1_line2_start', 'vp1_line2_end']
    vp2_names = ['vp2_line1_start', 'vp2_line1_end', 'vp2_line2_start', 'vp2_line2_end']

    from .properties import CONTROL_POINT_NAMES

    for i, name in enumerate(CONTROL_POINT_NAMES):
        if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
            continue

        pos = _pt(name)
        is_hover = (i == hover_idx)

        if name in vp1_names:
            col = COL_VP1_HANDLE_HOVER if is_hover else COL_VP1_HANDLE
        elif name in vp2_names:
            col = COL_VP2_HANDLE_HOVER if is_hover else COL_VP2_HANDLE
        elif name == 'origin_point':
            col = COL_ORIGIN_HOVER if is_hover else COL_ORIGIN
        else:
            col = COL_REF_HANDLE_HOVER if is_hover else COL_REF_HANDLE

        radius = HANDLE_HOVER_RADIUS if is_hover else HANDLE_RADIUS
        _draw_filled_circle(shader, pos[0], pos[1], radius, col)

    # --- Draw VP indicators ---
    _draw_vp_indicators(shader, props, frame)

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


def _draw_vp_indicators(shader, props, frame):
    """Draw small diamonds at the computed vanishing point positions."""
    # We need the solver result - compute it here for display
    cam = bpy.context.scene.camera
    if cam is None:
        return

    bg_images = cam.data.background_images
    if not bg_images:
        return

    # Get image aspect
    bg = bg_images[0]
    if bg.image is None:
        return

    w, h = bg.image.size
    if w == 0 or h == 0:
        return

    aspect = w / h

    # Compute VPs
    vp1_s1 = solv.relative_to_image_plane(props.vp1_line1_start[0], props.vp1_line1_start[1], aspect)
    vp1_e1 = solv.relative_to_image_plane(props.vp1_line1_end[0], props.vp1_line1_end[1], aspect)
    vp1_s2 = solv.relative_to_image_plane(props.vp1_line2_start[0], props.vp1_line2_start[1], aspect)
    vp1_e2 = solv.relative_to_image_plane(props.vp1_line2_end[0], props.vp1_line2_end[1], aspect)

    vp2_s1 = solv.relative_to_image_plane(props.vp2_line1_start[0], props.vp2_line1_start[1], aspect)
    vp2_e1 = solv.relative_to_image_plane(props.vp2_line1_end[0], props.vp2_line1_end[1], aspect)
    vp2_s2 = solv.relative_to_image_plane(props.vp2_line2_start[0], props.vp2_line2_start[1], aspect)
    vp2_e2 = solv.relative_to_image_plane(props.vp2_line2_end[0], props.vp2_line2_end[1], aspect)

    fu = solv.line_intersection(vp1_s1, vp1_e1, vp1_s2, vp1_e2)
    fv = solv.line_intersection(vp2_s1, vp2_e1, vp2_s2, vp2_e2)

    for vp, col in [(fu, COL_VP1_LINE), (fv, COL_VP2_LINE)]:
        if vp is None:
            continue

        # Convert back to relative then to screen
        rel = solv.image_plane_to_relative(vp[0], vp[1], aspect)
        sx, sy = normalized_to_screen(rel[0], rel[1], frame)

        # Only draw if on screen (roughly)
        region = bpy.context.region
        if -500 < sx < region.width + 500 and -500 < sy < region.height + 500:
            # Draw diamond
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
