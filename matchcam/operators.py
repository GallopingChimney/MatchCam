"""
Operators for MatchCam.

- MATCHCAM_OT_interact: Persistent modal for dragging control points
- MATCHCAM_OT_setup: Create camera, load background image
- MATCHCAM_OT_reset: Reset control points to defaults
- MATCHCAM_OT_enable: Toggle enable/disable
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

    pp = (props.principal_point[0], props.principal_point[1]) if props.use_custom_pp else (0.5, 0.5)

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
                # Update the dragged control point
                name = CONTROL_POINT_NAMES[self._drag_idx]
                # Skip inactive points
                if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
                    self._dragging = False
                    self._drag_idx = -1
                elif name in VP3_POINT_NAMES and props.mode != '3VP':
                    self._dragging = False
                    self._drag_idx = -1
                else:
                    if event.shift:
                        # Precision mode: 1/4 speed relative to last position
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

                    # Clamp to 0-1
                    nx = max(0.0, min(1.0, nx))
                    ny = max(0.0, min(1.0, ny))

                    # Ctrl+drag: constrain to horizontal or vertical
                    if event.ctrl:
                        dx = abs(mx - self._drag_start_screen[0])
                        dy = abs(my - self._drag_start_screen[1])
                        if dx >= dy:
                            ny = self._drag_start_value[1]
                        else:
                            nx = self._drag_start_value[0]

                    setattr(props, name, (nx, ny))

                    # Store loupe state for drawing
                    scene["_matchcam_precision"] = event.shift
                    scene["_matchcam_drag_screen"] = (mx, my)

                    # Run solver
                    _run_solver(scene)

                    # Redraw
                    self._tag_redraw(context)

                self._last_mouse_screen = (mx, my)
                return {'RUNNING_MODAL'}
            else:
                # Update hover state
                old_hover = self._hover_idx
                self._hover_idx = self._hit_test(props, mx, my, frame)
                scene["_matchcam_hover_idx"] = self._hover_idx

                if old_hover != self._hover_idx:
                    self._tag_redraw(context)

                return {'PASS_THROUGH'}

        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                hit = self._hit_test(props, mx, my, frame)
                if hit >= 0:
                    name = CONTROL_POINT_NAMES[hit]
                    # Skip ref points if disabled
                    if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
                        return {'PASS_THROUGH'}

                    self._dragging = True
                    self._drag_idx = hit
                    val = getattr(props, name)
                    self._drag_start_value = (val[0], val[1])
                    self._drag_start_screen = (mx, my)
                    self._last_mouse_screen = (mx, my)
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}

            elif event.value == 'RELEASE':
                if self._dragging:
                    # Push undo snapshot after completing a drag
                    self._undo_stack.append(_snapshot_points(props))
                    self._redo_stack.clear()
                    self._dragging = False
                    self._drag_idx = -1
                    scene["_matchcam_precision"] = False
                    self._tag_redraw(context)
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}

        elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            if self._dragging and self._drag_idx >= 0:
                # Cancel drag - restore original value (no undo push)
                name = CONTROL_POINT_NAMES[self._drag_idx]
                setattr(props, name, self._drag_start_value)
                _run_solver(scene)
                self._dragging = False
                self._drag_idx = -1
                scene["_matchcam_precision"] = False
                self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        elif event.type == 'ESC' and event.value == 'PRESS':
            if self._dragging and self._drag_idx >= 0:
                # Cancel drag - restore original value (no undo push)
                name = CONTROL_POINT_NAMES[self._drag_idx]
                setattr(props, name, self._drag_start_value)
                _run_solver(scene)
                self._dragging = False
                self._drag_idx = -1
                scene["_matchcam_precision"] = False
                self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # --- Undo / Redo ---
        elif event.type == 'Z' and event.value == 'PRESS' and event.ctrl:
            if event.shift:
                # Redo
                if self._redo_stack:
                    self._undo_stack.append(_snapshot_points(props))
                    snap = self._redo_stack.pop()
                    _restore_snapshot(props, snap)
                    _run_solver(scene)
                    self._tag_redraw(context)
                return {'RUNNING_MODAL'}
            else:
                # Undo
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

        for i, name in enumerate(CONTROL_POINT_NAMES):
            if name in VP3_POINT_NAMES and not is_3vp:
                continue
            if name in ('ref_point_a', 'ref_point_b') and not props.ref_distance_enabled:
                continue

            val = getattr(props, name)
            sx, sy = normalized_to_screen(val[0], val[1], frame)

            dist = math.sqrt((mx - sx) ** 2 + (my - sy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _tag_redraw(self, context):
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    def _cleanup(self, context):
        unregister_draw_handler()
        context.scene["_matchcam_hover_idx"] = -1
        context.scene["_matchcam_precision"] = False
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

        # Create camera if needed
        if scene.camera is None:
            cam_data = bpy.data.cameras.new("MatchCam Camera")
            cam_obj = bpy.data.objects.new("MatchCam Camera", cam_data)
            scene.collection.objects.link(cam_obj)
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
        bg.alpha = 1.0
        bg.display_depth = 'BACK'

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
# Reset operator
# ---------------------------------------------------------------------------

class MATCHCAM_OT_reset(bpy.types.Operator):
    """Reset all control points to default positions"""
    bl_idname = "matchcam.reset"
    bl_label = "Reset Control Points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.matchcam
        for name, default in CONTROL_POINT_DEFAULTS.items():
            setattr(props, name, default)

        # Re-run solver
        if props.enabled:
            _run_solver(context.scene)

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    MATCHCAM_OT_interact,
    MATCHCAM_OT_enable,
    MATCHCAM_OT_setup,
    MATCHCAM_OT_reset,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    unregister_draw_handler()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
