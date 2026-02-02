"""
Operators for MatchCam.

- MATCHCAM_OT_interact: Persistent modal for dragging control points
- MATCHCAM_OT_setup: Create camera, load background image
- MATCHCAM_OT_reset: Reset control points to defaults
- MATCHCAM_OT_enable: Toggle enable/disable
- MATCHCAM_OT_lock_camera: Toggle camera transform locks
- MATCHCAM_OT_keyframe_camera: Insert keyframes on camera
"""

from __future__ import annotations

import math

import bpy
from bpy.props import StringProperty

from .drawing import (
    get_camera_frame_px,
    normalized_to_screen,
    screen_to_normalized,
    register_draw_handler,
    unregister_draw_handler,
    HANDLE_RADIUS,
)
from .properties import CONTROL_POINT_NAMES, CONTROL_POINT_DEFAULTS, VP3_POINT_NAMES
from . import solver as solv


# ---------------------------------------------------------------------------
# Snapshot helpers (for undo/redo)
# ---------------------------------------------------------------------------

def _snapshot_points(props) -> dict:
    """Capture all control point values as a dict."""
    return {name: (getattr(props, name)[0], getattr(props, name)[1])
            for name in CONTROL_POINT_NAMES}


def _restore_snapshot(props, snap: dict):
    """Restore all control point values from a snapshot."""
    for name, val in snap.items():
        setattr(props, name, val)


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------

def _get_image_aspect(scene) -> float | None:
    """Get the aspect ratio from the camera's background image."""
    cam = scene.camera
    if cam is None:
        return None

    bg_images = cam.data.background_images
    if not bg_images:
        return None

    bg = bg_images[0]
    if bg.image is None:
        return None

    w, h = bg.image.size
    if w == 0 or h == 0:
        return None

    return w / h


def _run_solver(scene):
    """Run the solver and apply results to the camera."""
    props = scene.matchcam
    cam = scene.camera
    if cam is None:
        return

    aspect = _get_image_aspect(scene)
    if aspect is None:
        return

    pp = (props.principal_point[0], props.principal_point[1]) if (props.use_custom_pp and props.mode == '2VP') else (0.5, 0.5)

    # Build solver kwargs
    solver_kwargs = dict(
        vp1_l1_start=(props.vp1_line1_start[0], props.vp1_line1_start[1]),
        vp1_l1_end=(props.vp1_line1_end[0], props.vp1_line1_end[1]),
        vp1_l2_start=(props.vp1_line2_start[0], props.vp1_line2_start[1]),
        vp1_l2_end=(props.vp1_line2_end[0], props.vp1_line2_end[1]),
        vp2_l1_start=(props.vp2_line1_start[0], props.vp2_line1_start[1]),
        vp2_l1_end=(props.vp2_line1_end[0], props.vp2_line1_end[1]),
        vp2_l2_start=(props.vp2_line2_start[0], props.vp2_line2_start[1]),
        vp2_l2_end=(props.vp2_line2_end[0], props.vp2_line2_end[1]),
        origin=(props.origin_point[0], props.origin_point[1]),
        vp1_axis=props.vp1_axis,
        vp2_axis=props.vp2_axis,
        image_aspect=aspect,
        sensor_width=cam.data.sensor_width,
        principal_point=pp,
        ref_distance_enabled=props.ref_distance_enabled,
        ref_distance=props.ref_distance,
        ref_point_a=(props.ref_point_a[0], props.ref_point_a[1]),
        ref_point_b=(props.ref_point_b[0], props.ref_point_b[1]),
    )

    # Add 3VP lines if in 3VP mode
    if props.mode == '3VP':
        solver_kwargs.update(
            vp3_l1_start=(props.vp3_line1_start[0], props.vp3_line1_start[1]),
            vp3_l1_end=(props.vp3_line1_end[0], props.vp3_line1_end[1]),
            vp3_l2_start=(props.vp3_line2_start[0], props.vp3_line2_start[1]),
            vp3_l2_end=(props.vp3_line2_end[0], props.vp3_line2_end[1]),
        )

    result = solv.solve_2vp(**solver_kwargs)

    if result is None:
        # Store invalid state for UI feedback
        scene['_matchcam_valid'] = False
        return

    scene['_matchcam_valid'] = True
    scene['_matchcam_focal_mm'] = result.focal_length_mm
    scene['_matchcam_hfov'] = result.hfov_deg
    scene['_matchcam_vfov'] = result.vfov_deg

    # Apply to camera
    cam.data.lens = result.focal_length_mm

    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = result.rotation_quaternion

    cam.location = result.location

    # Apply camera shift (from principal point offset, significant in 3VP mode)
    cam.data.shift_x = result.shift_x
    cam.data.shift_y = result.shift_y


# ---------------------------------------------------------------------------
# VP intersection helpers
# ---------------------------------------------------------------------------

_VP_DIAMOND_HIT_RADIUS = 12.0

_VP_PREFIXES = ('vp1', 'vp2', 'vp3')


def _line_intersect_2d(s1, e1, s2, e2):
    """Intersect two infinite 2D lines. Returns (x, y) or None if parallel."""
    d1x, d1y = e1[0] - s1[0], e1[1] - s1[1]
    d2x, d2y = e2[0] - s2[0], e2[1] - s2[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-10:
        return None
    t = ((s2[0] - s1[0]) * d2y - (s2[1] - s1[1]) * d2x) / denom
    return (s1[0] + t * d1x, s1[1] + t * d1y)


def _get_vp_normalized(props, vp_group):
    """Compute VP intersection in normalized coords for a VP group (0-2)."""
    prefix = _VP_PREFIXES[vp_group]
    s1 = getattr(props, f"{prefix}_line1_start")
    e1 = getattr(props, f"{prefix}_line1_end")
    s2 = getattr(props, f"{prefix}_line2_start")
    e2 = getattr(props, f"{prefix}_line2_end")
    return _line_intersect_2d(
        (s1[0], s1[1]), (e1[0], e1[1]),
        (s2[0], s2[1]), (e2[0], e2[1]),
    )


def _move_vp_to(props, vp_group, target):
    """Adjust line endpoints so VP group converges at *target* (normalized).

    The handle farthest from the target VP stays fixed as a pivot; the
    nearer handle is moved onto the line from pivot through target,
    preserving the segment length.
    """
    prefix = _VP_PREFIXES[vp_group]
    for line_num in (1, 2):
        s_name = f"{prefix}_line{line_num}_start"
        e_name = f"{prefix}_line{line_num}_end"
        s = getattr(props, s_name)
        e = getattr(props, e_name)
        s = (s[0], s[1])
        e = (e[0], e[1])

        seg_dx, seg_dy = e[0] - s[0], e[1] - s[1]
        seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        if seg_len < 1e-6:
            continue

        # Determine which handle is farther from the VP target (= pivot)
        ds = math.sqrt((s[0] - target[0]) ** 2 + (s[1] - target[1]) ** 2)
        de = math.sqrt((e[0] - target[0]) ** 2 + (e[1] - target[1]) ** 2)

        if ds >= de:
            # Start is farther — it stays fixed, end moves
            pivot = s
            move_name = e_name
        else:
            # End is farther — it stays fixed, start moves
            pivot = e
            move_name = s_name

        # Direction from target VP through pivot
        vdx, vdy = pivot[0] - target[0], pivot[1] - target[1]
        vlen = math.sqrt(vdx * vdx + vdy * vdy)
        if vlen < 1e-6:
            continue

        ndx, ndy = vdx / vlen, vdy / vlen
        # Place the moving handle at segment length from pivot, toward the VP
        setattr(props, move_name, (pivot[0] - ndx * seg_len, pivot[1] - ndy * seg_len))


def _pivot_adjust_partner(vp, dragged_pos, partner_pos):
    """Compute new partner handle position when pivoting around a locked VP.

    The partner handle is placed on the line from *vp* through *dragged_pos*,
    preserving its original distance from *vp*.

    Returns (nx, ny) for the partner, or None if degenerate.
    """
    dx, dy = dragged_pos[0] - vp[0], dragged_pos[1] - vp[1]
    dist_d = math.sqrt(dx * dx + dy * dy)
    if dist_d < 1e-10:
        return None

    # Direction from VP toward dragged handle
    ndx, ndy = dx / dist_d, dy / dist_d

    # Partner's original distance from VP
    px, py = partner_pos[0] - vp[0], partner_pos[1] - vp[1]
    dist_p = math.sqrt(px * px + py * py)

    # Determine if partner was on the same side of VP as dragged handle
    dot = px * ndx + py * ndy
    sign = 1.0 if dot >= 0 else -1.0

    return (vp[0] + ndx * dist_p * sign, vp[1] + ndy * dist_p * sign)


def _get_vp_group_for_handle(drag_idx):
    """Return (vp_group, line_num, partner_name) for a VP handle index, or None."""
    if drag_idx < 0 or drag_idx > 11:
        return None
    vp_group = drag_idx // 4  # 0, 1, or 2
    local = drag_idx % 4      # 0-3 within group
    line_num = (local // 2) + 1  # 1 or 2
    is_start = (local % 2 == 0)
    prefix = _VP_PREFIXES[vp_group]
    if is_start:
        partner_name = f"{prefix}_line{line_num}_end"
    else:
        partner_name = f"{prefix}_line{line_num}_start"
    return (vp_group, line_num, partner_name)


# ---------------------------------------------------------------------------
# Persistent modal operator
# ---------------------------------------------------------------------------

class MATCHCAM_OT_interact(bpy.types.Operator):
    """Interactive control point adjustment for MatchCam"""
    bl_idname = "matchcam.interact"
    bl_label = "MatchCam Interact"
    bl_options = {'INTERNAL'}

    _dragging: bool = False
    _drag_idx: int = -1
    _drag_start_value: tuple[float, float] = (0, 0)
    _drag_start_screen: tuple[float, float] = (0, 0)
    _last_mouse_screen: tuple[float, float] = (0, 0)
    _hover_idx: int = -1
    _undo_stack: list = []
    _redo_stack: list = []

    # VP intersection dragging
    _vp_dragging: bool = False
    _vp_group: int = -1
    _vp_drag_start_snap: dict = {}

    # Alt pivot mode: locked VP intersection
    _pivot_vp: tuple | None = None

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        register_draw_handler()

        context.window_manager.modal_handler_add(self)

        # Initialize undo stack with current state
        self._undo_stack = [_snapshot_points(context.scene.matchcam)]
        self._redo_stack = []

        # Run solver once on start
        _run_solver(context.scene)

        # Tag redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        # Check if we should stop
        if not hasattr(scene, 'matchcam') or not scene.matchcam.enabled:
            self._cleanup(context)
            return {'CANCELLED'}

        props = scene.matchcam
        frame = get_camera_frame_px(context)

        if frame is None:
            # Not in camera view - auto-return to camera view so the user
            # doesn't lose the overlay after accidentally orbiting away.
            if scene.camera is not None:
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                if space.region_3d.view_perspective != 'CAMERA':
                                    space.region_3d.view_perspective = 'CAMERA'
                        break
            return {'PASS_THROUGH'}

        mx, my = event.mouse_region_x, event.mouse_region_y

        if event.type == 'MOUSEMOVE':
            if self._dragging and self._drag_idx >= 0:
                # --- Regular control point drag ---
                name = CONTROL_POINT_NAMES[self._drag_idx]
                # Skip inactive points
                if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
                    self._dragging = False
                    self._drag_idx = -1
                elif name in VP3_POINT_NAMES and props.mode != '3VP':
                    self._dragging = False
                    self._drag_idx = -1
                elif name == 'principal_point' and not (props.mode == '2VP' and props.use_custom_pp):
                    self._dragging = False
                    self._drag_idx = -1
                else:
                    if event.shift:
                        PRECISION_FACTOR = 0.25
                        last_nx, last_ny = screen_to_normalized(
                            self._last_mouse_screen[0],
                            self._last_mouse_screen[1], frame)
                        full_nx, full_ny = screen_to_normalized(mx, my, frame)
                        cur = getattr(props, name)
                        nx = cur[0] + (full_nx - last_nx) * PRECISION_FACTOR
                        ny = cur[1] + (full_ny - last_ny) * PRECISION_FACTOR
                    else:
                        nx, ny = screen_to_normalized(mx, my, frame)

                    # Clamp to 0-1 (except origin_point which can go outside frame)
                    if name != 'origin_point':
                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))

                    # Alt+drag on VP handle: pivot around locked VP
                    pivot_info = _get_vp_group_for_handle(self._drag_idx)
                    if event.alt and pivot_info is not None:
                        vp_group, _line_num, partner_name = pivot_info
                        # Lock the VP intersection on first Ctrl+Shift frame
                        if self._pivot_vp is None:
                            self._pivot_vp = _get_vp_normalized(props, vp_group)
                        if self._pivot_vp is not None:
                            partner_val = getattr(props, partner_name)
                            partner_old = (partner_val[0], partner_val[1])
                            new_partner = _pivot_adjust_partner(
                                self._pivot_vp, (nx, ny), partner_old)
                            if new_partner is not None:
                                setattr(props, partner_name, new_partner)
                    else:
                        # Clear pivot lock when Alt not held
                        self._pivot_vp = None
                        # Ctrl+drag: constrain to H or V
                        if event.ctrl:
                            dx = abs(mx - self._drag_start_screen[0])
                            dy = abs(my - self._drag_start_screen[1])
                            if dx >= dy:
                                ny = self._drag_start_value[1]
                            else:
                                nx = self._drag_start_value[0]

                    setattr(props, name, (nx, ny))

                    # Store loupe state for drawing (centered on handle, not mouse)
                    handle_sx, handle_sy = normalized_to_screen(nx, ny, frame)
                    scene["_matchcam_precision"] = event.shift
                    scene["_matchcam_drag_screen"] = (handle_sx, handle_sy)
                    scene["_matchcam_drag_idx"] = self._drag_idx

                    _run_solver(scene)
                    self._tag_redraw(context)

                self._last_mouse_screen = (mx, my)
                return {'RUNNING_MODAL'}

            elif self._vp_dragging and self._vp_group >= 0:
                # --- VP intersection drag ---
                if event.shift:
                    PRECISION_FACTOR = 0.25
                    last_nx, last_ny = screen_to_normalized(
                        self._last_mouse_screen[0],
                        self._last_mouse_screen[1], frame)
                    full_nx, full_ny = screen_to_normalized(mx, my, frame)
                    cur_vp = _get_vp_normalized(props, self._vp_group)
                    if cur_vp:
                        nx = cur_vp[0] + (full_nx - last_nx) * PRECISION_FACTOR
                        ny = cur_vp[1] + (full_ny - last_ny) * PRECISION_FACTOR
                    else:
                        nx, ny = screen_to_normalized(mx, my, frame)
                else:
                    nx, ny = screen_to_normalized(mx, my, frame)

                _move_vp_to(props, self._vp_group, (nx, ny))

                # Store loupe/precision state
                vp_sx, vp_sy = normalized_to_screen(nx, ny, frame)
                scene["_matchcam_precision"] = event.shift
                scene["_matchcam_drag_screen"] = (vp_sx, vp_sy)
                # Use first handle index of the VP group for loupe color
                scene["_matchcam_drag_idx"] = self._vp_group * 4

                _run_solver(scene)
                self._tag_redraw(context)
                self._last_mouse_screen = (mx, my)
                return {'RUNNING_MODAL'}

            else:
                # --- Hover state ---
                old_hover = self._hover_idx
                self._hover_idx = self._hit_test(props, mx, my, frame)
                scene["_matchcam_hover_idx"] = self._hover_idx

                # VP hover
                old_vp_hover = scene.get("_matchcam_vp_hover", -1)
                if self._hover_idx < 0:
                    vp_hover = self._vp_hit_test(props, mx, my, frame)
                else:
                    vp_hover = -1
                scene["_matchcam_vp_hover"] = vp_hover

                # Show loupe when Shift is held over a hovered handle
                old_precision = scene.get("_matchcam_precision", False)
                if event.shift and self._hover_idx >= 0:
                    val = getattr(props, CONTROL_POINT_NAMES[self._hover_idx])
                    hsx, hsy = normalized_to_screen(val[0], val[1], frame)
                    scene["_matchcam_precision"] = True
                    scene["_matchcam_drag_screen"] = (hsx, hsy)
                    scene["_matchcam_drag_idx"] = self._hover_idx
                elif event.shift and vp_hover >= 0:
                    vp = _get_vp_normalized(props, vp_hover)
                    if vp is not None:
                        vsx, vsy = normalized_to_screen(vp[0], vp[1], frame)
                        scene["_matchcam_precision"] = True
                        scene["_matchcam_drag_screen"] = (vsx, vsy)
                        scene["_matchcam_drag_idx"] = vp_hover * 4
                    else:
                        scene["_matchcam_precision"] = False
                else:
                    scene["_matchcam_precision"] = False

                need_redraw = (old_hover != self._hover_idx
                               or old_vp_hover != vp_hover
                               or old_precision != scene.get("_matchcam_precision", False))
                if need_redraw:
                    self._tag_redraw(context)

                return {'PASS_THROUGH'}

        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                hit = self._hit_test(props, mx, my, frame)
                if hit >= 0:
                    name = CONTROL_POINT_NAMES[hit]
                    if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
                        return {'PASS_THROUGH'}
                    if name == 'principal_point' and not (props.mode == '2VP' and props.use_custom_pp):
                        return {'PASS_THROUGH'}

                    self._dragging = True
                    self._drag_idx = hit
                    val = getattr(props, name)
                    self._drag_start_value = (val[0], val[1])
                    self._drag_start_screen = (mx, my)
                    self._last_mouse_screen = (mx, my)
                    scene["_matchcam_drag_idx"] = hit
                    self._tag_redraw(context)
                    return {'RUNNING_MODAL'}

                # Check VP diamond hit
                vp_hit = self._vp_hit_test(props, mx, my, frame)
                if vp_hit >= 0:
                    self._vp_dragging = True
                    self._vp_group = vp_hit
                    self._vp_drag_start_snap = _snapshot_points(props)
                    self._last_mouse_screen = (mx, my)
                    scene["_matchcam_vp_drag"] = vp_hit
                    self._tag_redraw(context)
                    return {'RUNNING_MODAL'}

                return {'PASS_THROUGH'}

            elif event.value == 'RELEASE':
                if self._dragging:
                    self._undo_stack.append(_snapshot_points(props))
                    self._redo_stack.clear()
                    self._dragging = False
                    self._drag_idx = -1
                    self._pivot_vp = None
                    scene["_matchcam_precision"] = False
                    scene["_matchcam_drag_idx"] = -1
                    self._tag_redraw(context)
                    return {'RUNNING_MODAL'}
                if self._vp_dragging:
                    self._undo_stack.append(_snapshot_points(props))
                    self._redo_stack.clear()
                    self._vp_dragging = False
                    self._vp_group = -1
                    scene["_matchcam_precision"] = False
                    scene["_matchcam_drag_idx"] = -1
                    scene["_matchcam_vp_drag"] = -1
                    self._tag_redraw(context)
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}

        elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            if self._dragging and self._drag_idx >= 0:
                name = CONTROL_POINT_NAMES[self._drag_idx]
                setattr(props, name, self._drag_start_value)
                _run_solver(scene)
                self._dragging = False
                self._drag_idx = -1
                self._pivot_vp = None
                scene["_matchcam_precision"] = False
                scene["_matchcam_drag_idx"] = -1
                self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            if self._vp_dragging:
                _restore_snapshot(props, self._vp_drag_start_snap)
                _run_solver(scene)
                self._vp_dragging = False
                self._vp_group = -1
                scene["_matchcam_precision"] = False
                scene["_matchcam_drag_idx"] = -1
                scene["_matchcam_vp_drag"] = -1
                self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        elif event.type == 'ESC' and event.value == 'PRESS':
            if self._dragging and self._drag_idx >= 0:
                name = CONTROL_POINT_NAMES[self._drag_idx]
                setattr(props, name, self._drag_start_value)
                _run_solver(scene)
                self._dragging = False
                self._drag_idx = -1
                self._pivot_vp = None
                scene["_matchcam_precision"] = False
                scene["_matchcam_drag_idx"] = -1
                self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            if self._vp_dragging:
                _restore_snapshot(props, self._vp_drag_start_snap)
                _run_solver(scene)
                self._vp_dragging = False
                self._vp_group = -1
                scene["_matchcam_precision"] = False
                scene["_matchcam_drag_idx"] = -1
                scene["_matchcam_vp_drag"] = -1
                self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # --- Shift press/release: show/dismiss loupe ---
        elif event.type in ('LEFT_SHIFT', 'RIGHT_SHIFT'):
            if event.value == 'PRESS':
                if self._dragging and self._drag_idx >= 0:
                    # Already dragging a handle — show loupe at handle position
                    name = CONTROL_POINT_NAMES[self._drag_idx]
                    val = getattr(props, name)
                    hsx, hsy = normalized_to_screen(val[0], val[1], frame)
                    scene["_matchcam_precision"] = True
                    scene["_matchcam_drag_screen"] = (hsx, hsy)
                    scene["_matchcam_drag_idx"] = self._drag_idx
                    self._tag_redraw(context)
                elif self._vp_dragging and self._vp_group >= 0:
                    # Dragging VP diamond — show loupe at VP position
                    vp = _get_vp_normalized(props, self._vp_group)
                    if vp is not None:
                        vsx, vsy = normalized_to_screen(vp[0], vp[1], frame)
                        scene["_matchcam_precision"] = True
                        scene["_matchcam_drag_screen"] = (vsx, vsy)
                        scene["_matchcam_drag_idx"] = self._vp_group * 4
                        self._tag_redraw(context)
                elif self._hover_idx >= 0:
                    # Hovering a handle — show loupe at handle position
                    val = getattr(props, CONTROL_POINT_NAMES[self._hover_idx])
                    hsx, hsy = normalized_to_screen(val[0], val[1], frame)
                    scene["_matchcam_precision"] = True
                    scene["_matchcam_drag_screen"] = (hsx, hsy)
                    scene["_matchcam_drag_idx"] = self._hover_idx
                    self._tag_redraw(context)
                else:
                    vp_hover = scene.get("_matchcam_vp_hover", -1)
                    if vp_hover >= 0:
                        vp = _get_vp_normalized(props, vp_hover)
                        if vp is not None:
                            vsx, vsy = normalized_to_screen(vp[0], vp[1], frame)
                            scene["_matchcam_precision"] = True
                            scene["_matchcam_drag_screen"] = (vsx, vsy)
                            scene["_matchcam_drag_idx"] = vp_hover * 4
                            self._tag_redraw(context)
            elif event.value == 'RELEASE':
                if scene.get("_matchcam_precision", False):
                    scene["_matchcam_precision"] = False
                    self._tag_redraw(context)
            return {'PASS_THROUGH'}

        # --- Undo / Redo ---
        elif event.type == 'Z' and event.value == 'PRESS' and event.ctrl:
            if event.shift:
                if self._redo_stack:
                    self._undo_stack.append(_snapshot_points(props))
                    snap = self._redo_stack.pop()
                    _restore_snapshot(props, snap)
                    _run_solver(scene)
                    self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            else:
                if len(self._undo_stack) > 1:
                    self._redo_stack.append(self._undo_stack.pop())
                    snap = self._undo_stack[-1]
                    _restore_snapshot(props, snap)
                    _run_solver(scene)
                    self._tag_redraw(context)
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _hit_test(self, props, mx, my, frame) -> int:
        """Find the closest control point within grab radius. Returns index or -1."""
        best_dist = HANDLE_RADIUS + 5.0
        best_idx = -1

        is_3vp = (props.mode == '3VP')
        pp_active = (props.mode == '2VP' and props.use_custom_pp)

        for i, name in enumerate(CONTROL_POINT_NAMES):
            if name in VP3_POINT_NAMES and not is_3vp:
                continue
            if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
                continue
            if name == 'principal_point' and not pp_active:
                continue

            val = getattr(props, name)
            sx, sy = normalized_to_screen(val[0], val[1], frame)

            dist = math.sqrt((mx - sx) ** 2 + (my - sy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _vp_hit_test(self, props, mx, my, frame) -> int:
        """Check if mouse is near a VP diamond. Returns group (0-2) or -1."""
        is_3vp = (props.mode == '3VP')
        groups = [0, 1] + ([2] if is_3vp else [])

        best_dist = _VP_DIAMOND_HIT_RADIUS
        best_group = -1

        for g in groups:
            vp = _get_vp_normalized(props, g)
            if vp is None:
                continue
            sx, sy = normalized_to_screen(vp[0], vp[1], frame)
            # Only test VPs that are reasonably on-screen
            region = bpy.context.region
            if not (-500 < sx < region.width + 500 and -500 < sy < region.height + 500):
                continue
            dist = math.sqrt((mx - sx) ** 2 + (my - sy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_group = g

        return best_group

    def _tag_redraw(self, context):
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    def _cleanup(self, context):
        unregister_draw_handler()
        context.scene["_matchcam_hover_idx"] = -1
        context.scene["_matchcam_precision"] = False
        context.scene["_matchcam_drag_idx"] = -1
        context.scene["_matchcam_vp_hover"] = -1
        context.scene["_matchcam_vp_drag"] = -1
        self._tag_redraw(context)

    def cancel(self, context):
        self._cleanup(context)


# ---------------------------------------------------------------------------
# Enable / disable toggle
# ---------------------------------------------------------------------------

class MATCHCAM_OT_enable(bpy.types.Operator):
    """Toggle MatchCam on/off"""
    bl_idname = "matchcam.enable"
    bl_label = "Enable MatchCam"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.matchcam
        props.enabled = not props.enabled

        if props.enabled:
            # Switch to camera view if not already
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            if space.region_3d.view_perspective != 'CAMERA':
                                bpy.ops.view3d.view_camera()
                    break

            # Start modal
            bpy.ops.matchcam.interact('INVOKE_DEFAULT')
        else:
            unregister_draw_handler()
            # Redraw
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Setup operator
# ---------------------------------------------------------------------------

class MATCHCAM_OT_setup(bpy.types.Operator):
    """Set up camera and load background image"""
    bl_idname = "matchcam.setup"
    bl_label = "Setup Camera + Load Image"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.exr;*.hdr", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        props = scene.matchcam

        # Determine target camera from selection
        target = props.target_camera

        if target == '__NEW__':
            cam_data = bpy.data.cameras.new("MatchCam Camera")
            cam_obj = bpy.data.objects.new("MatchCam Camera", cam_data)
            scene.collection.objects.link(cam_obj)
            scene.camera = cam_obj
        else:
            cam_obj = bpy.data.objects.get(target)
            if cam_obj is None or cam_obj.type != 'CAMERA':
                self.report({'ERROR'}, f"Camera '{target}' not found")
                return {'CANCELLED'}
            scene.camera = cam_obj

        cam = scene.camera
        cam_data = cam.data

        # Load image
        img = bpy.data.images.load(self.filepath)

        # Set up background image
        cam_data.show_background_images = True

        # Clear existing backgrounds
        while cam_data.background_images:
            cam_data.background_images.remove(cam_data.background_images[0])

        bg = cam_data.background_images.new()
        bg.image = img
        bg.alpha = props.bg_alpha
        bg.display_depth = props.bg_display_depth

        # Match render resolution to image
        w, h = img.size
        scene.render.resolution_x = w
        scene.render.resolution_y = h

        # Switch to camera view
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                        # Also show background images in viewport
                        space.overlay.show_overlays = True
                break

        self.report({'INFO'}, f"Loaded image: {img.name} ({w}x{h})")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Reset operators
# ---------------------------------------------------------------------------

class MATCHCAM_OT_reset(bpy.types.Operator):
    """Reset all control points to default positions"""
    bl_idname = "matchcam.reset"
    bl_label = "Reset Control Points"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.matchcam
        for name, default in CONTROL_POINT_DEFAULTS.items():
            setattr(props, name, default)

        # Re-run solver
        if props.enabled:
            _run_solver(context.scene)

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


class MATCHCAM_OT_reset_origin(bpy.types.Operator):
    """Reset origin point to default position (image center bottom)"""
    bl_idname = "matchcam.reset_origin"
    bl_label = "Reset Origin Point"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.matchcam
        setattr(props, 'origin_point', CONTROL_POINT_DEFAULTS['origin_point'])

        if props.enabled:
            _run_solver(context.scene)

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Lock camera
# ---------------------------------------------------------------------------

class MATCHCAM_OT_lock_camera(bpy.types.Operator):
    """Toggle camera transform locks (location, rotation)"""
    bl_idname = "matchcam.lock_camera"
    bl_label = "Lock Camera"
    bl_options = {'REGISTER'}

    def execute(self, context):
        cam = context.scene.camera
        if cam is None:
            self.report({'WARNING'}, "No camera")
            return {'CANCELLED'}

        is_locked = all(cam.lock_location) and all(cam.lock_rotation)
        new_state = not is_locked

        cam.lock_location = (new_state, new_state, new_state)
        cam.lock_rotation = (new_state, new_state, new_state)
        cam.lock_rotation_w = new_state
        cam.lock_scale = (new_state, new_state, new_state)

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Keyframe camera
# ---------------------------------------------------------------------------

class MATCHCAM_OT_keyframe_camera(bpy.types.Operator):
    """Insert keyframes for camera location, rotation, focal length, and shift"""
    bl_idname = "matchcam.keyframe_camera"
    bl_label = "Keyframe Camera"
    bl_options = {'REGISTER'}

    def execute(self, context):
        cam = context.scene.camera
        if cam is None:
            self.report({'WARNING'}, "No camera")
            return {'CANCELLED'}

        cam.keyframe_insert(data_path="location")
        cam.keyframe_insert(data_path="rotation_quaternion")
        cam.data.keyframe_insert(data_path="lens")
        cam.data.keyframe_insert(data_path="shift_x")
        cam.data.keyframe_insert(data_path="shift_y")

        self.report({'INFO'}, f"Keyframed camera at frame {context.scene.frame_current}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    MATCHCAM_OT_interact,
    MATCHCAM_OT_enable,
    MATCHCAM_OT_setup,
    MATCHCAM_OT_reset,
    MATCHCAM_OT_reset_origin,
    MATCHCAM_OT_lock_camera,
    MATCHCAM_OT_keyframe_camera,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    unregister_draw_handler()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
