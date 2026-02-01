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
        icon = 'PAUSE' if props.enabled else 'PLAY'
        text = "Disable" if props.enabled else "Enable"
        row.operator("matchcam.enable", text=text, icon=icon, depress=props.enabled)

        layout.separator()

        # Setup
        layout.operator("matchcam.setup", icon='IMAGE_DATA')

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

        # Principal point
        box = layout.box()
        box.prop(props, "use_custom_pp", text="Custom Principal Point")
        if props.use_custom_pp:
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
            box.label(text="No camera - run Setup first")
        elif not is_valid:
            box.label(text="Invalid configuration", icon='ERROR')
            box.label(text="Adjust lines so they converge")
        else:
            focal = scene.get('_matchcam_focal_mm', 0)
            hfov = scene.get('_matchcam_hfov', 0)
            vfov = scene.get('_matchcam_vfov', 0)

            box.label(text=f"Focal Length: {focal:.1f} mm")
            box.label(text=f"FOV: {hfov:.1f}\u00b0 x {vfov:.1f}\u00b0")


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
