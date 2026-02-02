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
        box.label(text="Camera", icon='CAMERA_DATA')

        row = box.row(align=True)
        row.label(text="Camera:")
        row.prop(props, "target_camera", text="")

        # Warn if selected camera has an existing background image
        target = props.target_camera
        if target != '__NEW__':
            cam_obj = bpy.data.objects.get(target)
            if cam_obj and cam_obj.type == 'CAMERA' and cam_obj.data.background_images:
                if any(bg.image for bg in cam_obj.data.background_images):
                    warn_row = box.row()
                    warn_row.alert = True
                    warn_row.label(text="Existing background will be replaced!", icon='ERROR')

        row = box.row()
        row.scale_y = 1.5
        row.operator("matchcam.setup", icon='IMAGE_DATA')

        # Background opacity
        box.prop(props, "bg_alpha", slider=True)

        layout.separator()

        # Mode: 2VP / 3VP
        box = layout.box()
        box.label(text="Mode", icon='OBJECT_DATA')
        row = box.row(align=True)
        row.prop(props, "mode", expand=True)

        if props.mode == '3VP':
            box.label(text="VP3 lines: blue", icon='INFO')
            box.label(text="Derives principal point + camera shift")

        layout.separator()

        # Axis assignment
        box = layout.box()
        box.label(text="Axis Assignment", icon='ORIENTATION_GLOBAL')
        row = box.row()
        row.prop(props, "vp1_axis", text="VP1")
        row.prop(props, "vp2_axis", text="VP2")

        # Warn if same axis
        if props.vp1_axis == props.vp2_axis:
            box.label(text="VP axes must differ!", icon='ERROR')

        layout.separator()

        # Principal point (only in 2VP mode - in 3VP it's auto-derived)
        if props.mode == '2VP':
            box = layout.box()
            box.prop(props, "use_custom_pp", text="Custom Optical Center")
            if props.use_custom_pp:
                box.label(text="Lens optical center on image", icon='INFO')
                row = box.row()
                row.prop(props, "principal_point", index=0, text="X")
                row.prop(props, "principal_point", index=1, text="Y")

        # Reference distance
        box = layout.box()
        box.prop(props, "ref_distance_enabled")
        if props.ref_distance_enabled:
            box.prop(props, "ref_distance")

        layout.separator()

        # Reset
        layout.operator("matchcam.reset", icon='FILE_REFRESH')

        layout.separator()

        # Status readout
        box = layout.box()
        box.label(text="Status", icon='INFO')

        is_valid = scene.get('_matchcam_valid', False)

        if not props.enabled:
            box.label(text="Disabled")
        elif scene.camera is None:
            row = box.row()
            row.alert = True
            row.label(text="No camera - run Setup first", icon='ERROR')
        elif not is_valid:
            row = box.row()
            row.alert = True
            row.label(text="Invalid configuration", icon='ERROR')
            row = box.row()
            row.alert = True
            row.label(text="Adjust lines so they converge")
        else:
            focal = scene.get('_matchcam_focal_mm', 0)
            hfov = scene.get('_matchcam_hfov', 0)
            vfov = scene.get('_matchcam_vfov', 0)

            box.label(text=f"Focal Length: {focal:.1f} mm")
            box.label(text=f"FOV: {hfov:.1f}\u00b0 x {vfov:.1f}\u00b0")

            # Show shift values in 3VP mode
            if props.mode == '3VP' and scene.camera is not None:
                cam = scene.camera.data
                if abs(cam.shift_x) > 0.001 or abs(cam.shift_y) > 0.001:
                    box.label(text=f"Shift: {cam.shift_x:.3f} / {cam.shift_y:.3f}")


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
