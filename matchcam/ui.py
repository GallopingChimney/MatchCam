"""
UI panel for MatchCam in the 3D Viewport sidebar (N-panel).
"""

from __future__ import annotations

import bpy


class MATCHCAM_PT_main(bpy.types.Panel):
    bl_label = "MatchCam"
    bl_idname = "MATCHCAM_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MatchCam"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.matchcam

        # Enable toggle
        row = layout.row()
        row.scale_y = 1.905
        icon = 'PAUSE' if props.enabled else 'PLAY'
        text = "Disable" if props.enabled else "Enable"
        row.operator("matchcam.enable", text=text, icon=icon, depress=props.enabled)

        layout.separator()

        # Camera section
        box = layout.box()
        box.label(text="Camera", icon='OUTLINER_DATA_CAMERA')

        row = box.row(align=True)
        split = row.split(factor=0.25, align=True)
        split.label(text="Camera:")
        split.prop(props, "target_camera", text="")

        # Warn if selected camera has an existing background image
        target = props.target_camera
        if target != '__NEW__':
            cam_obj = bpy.data.objects.get(target)
            if cam_obj and cam_obj.type == 'CAMERA' and cam_obj.data.background_images:
                if any(bg.image for bg in cam_obj.data.background_images):
                    warn_row = box.row()
                    warn_row.alert = True
                    warn_row.label(text="Existing background will be replaced!", icon='ERROR')

        # Image opacity + fill toggle
        row = box.row(align=True)
        row.prop(props, "bg_alpha", text="Image Opacity", slider=True)
        row.prop(props, "show_fill", text="", icon='MESH_PLANE')

        # Front / Back display depth
        row = box.row(align=True)
        row.prop(props, "bg_display_depth", expand=True)

        row = box.row()
        row.scale_y = 1.905
        row.operator("matchcam.setup", icon='IMAGE_DATA')

        layout.separator()

        # Mode + Axis assignment
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "mode", expand=True)

        row = box.row(align=True)
        row.label(text="VP1")
        row.prop(props, "vp1_axis", expand=True)
        row = box.row(align=True)
        row.label(text="Up" if props.mode == '1VP' else "VP2")
        row.prop(props, "vp2_axis", expand=True)

        # Warn if same axis
        if props.vp1_axis == props.vp2_axis:
            warn_row = box.row()
            warn_row.alert = True
            warn_row.label(text="VP axes must differ!", icon='ERROR')

        # Focal length (1VP only)
        if props.mode == '1VP':
            box.prop(props, "focal_length_1vp")

        # Optical centre + Reference distance (same box)
        if props.mode in ('1VP', '2VP'):
            box.prop(props, "use_custom_pp", text="Custom Optical Center")
            if props.use_custom_pp:
                row = box.row()
                row.prop(props, "principal_point", index=0, text="X")
                row.prop(props, "principal_point", index=1, text="Y")

        box.prop(props, "ref_distance_enabled")
        if props.ref_distance_enabled:
            box.prop(props, "ref_distance")

        layout.separator()

        # Action buttons (icon-only, full width)
        has_cam = scene.camera is not None
        n_cols = 4 if has_cam else 2
        grid = layout.grid_flow(row_major=True, columns=n_cols, even_columns=True, align=True)
        grid.operator("matchcam.reset", text="", icon='FILE_REFRESH')
        grid.operator("matchcam.reset_origin", text="", icon='OBJECT_ORIGIN')
        if has_cam:
            is_locked = all(scene.camera.lock_location) and all(scene.camera.lock_rotation)
            lock_icon = 'LOCKED' if is_locked else 'UNLOCKED'
            grid.operator("matchcam.lock_camera", text="", icon=lock_icon, depress=is_locked)
            grid.operator("matchcam.keyframe_camera", text="", icon='KEYTYPE_KEYFRAME_VEC')

        layout.separator()

        # Status readout
        box = layout.box()
        box.label(text="Status", icon='INFO')

        is_valid = scene.get('_matchcam_valid', False)
        col = box.column(align=True)

        if not props.enabled:
            col.label(text="Disabled")
        elif scene.camera is None:
            row = col.row()
            row.alert = True
            row.label(text="No camera - run Setup first", icon='ERROR')
        elif not scene.camera.data.background_images or not any(
                bg.image for bg in scene.camera.data.background_images):
            row = col.row()
            row.alert = True
            row.label(text="No background image - run Setup", icon='ERROR')
        elif not is_valid:
            row = col.row()
            row.alert = True
            row.label(text="Invalid configuration", icon='ERROR')
            row = col.row()
            row.alert = True
            if props.mode == '3VP':
                row.label(text="Adjust VP1/VP2/VP3 lines to converge")
            elif props.mode == '1VP':
                row.label(text="Adjust VP1 lines to converge")
            else:
                row.label(text="Adjust VP1/VP2 lines to converge")
        else:
            col.label(text=f"Mode: {props.mode}")

            focal = scene.get('_matchcam_focal_mm', 0)
            hfov = scene.get('_matchcam_hfov', 0)
            vfov = scene.get('_matchcam_vfov', 0)

            col.label(text=f"Focal Length: {focal:.1f} mm")
            col.label(text=f"FOV: {hfov:.1f}\u00b0 x {vfov:.1f}\u00b0")

            # Show shift values in 3VP mode
            if props.mode == '3VP' and scene.camera is not None:
                cam = scene.camera.data
                if abs(cam.shift_x) > 0.001 or abs(cam.shift_y) > 0.001:
                    col.label(text=f"Shift: {cam.shift_x:.3f} / {cam.shift_y:.3f}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    MATCHCAM_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
