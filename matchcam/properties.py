import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
)


def _on_solver_property_update(self, context):
    """Re-run solver when a solver-affecting property changes."""
    from .operators import _run_solver
    if self.enabled:
        _run_solver(context.scene)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _on_ref_distance_update(self, context):
    """Re-run solver when reference distance slider changes."""
    from .operators import _run_solver
    if self.enabled and self.ref_distance_enabled:
        _run_solver(context.scene)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _on_bg_alpha_update(self, context):
    """Update camera background image opacity in real time."""
    cam = context.scene.camera
    if cam and cam.data.background_images:
        for bg in cam.data.background_images:
            bg.alpha = self.bg_alpha
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _on_bg_depth_update(self, context):
    """Update camera background image display depth."""
    cam = context.scene.camera
    if cam and cam.data.background_images:
        for bg in cam.data.background_images:
            bg.display_depth = self.bg_display_depth
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _on_overlay_update(self, context):
    """Redraw viewport when an overlay setting changes."""
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


_camera_items_ref = []


def _get_camera_items(self, context):
    """Dynamic enum items: list existing cameras + 'New Camera' option."""
    global _camera_items_ref
    items = []
    scene_cam = context.scene.camera
    idx = 0

    # Current scene camera first (if any)
    if scene_cam and scene_cam.type == 'CAMERA':
        has_bg = bool(scene_cam.data.background_images and
                      any(bg.image for bg in scene_cam.data.background_images))
        desc = "Active camera (has background)" if has_bg else "Active camera"
        items.append((scene_cam.name, scene_cam.name, desc, 'CAMERA_DATA', idx))
        idx += 1

    # Other cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' and obj != scene_cam:
            has_bg = bool(obj.data.background_images and
                          any(bg.image for bg in obj.data.background_images))
            desc = "Has background image" if has_bg else "No background image"
            items.append((obj.name, obj.name, desc, 'CAMERA_DATA', idx))
            idx += 1

    # New camera option at end
    items.append(('__NEW__', "New Camera", "Create a new camera", 'ADD', idx))

    _camera_items_ref = items
    return _camera_items_ref


class MatchCamProperties(bpy.types.PropertyGroup):
    """Properties for MatchCam camera matching."""

    enabled: BoolProperty(
        name="Enable",
        description="Enable MatchCam overlay and interaction",
        default=False,
    )

    # Mode: 1VP, 2VP or 3VP
    mode: EnumProperty(
        name="Mode",
        description="Number of vanishing points to use",
        items=[
            ('1VP', "1-Point", "One vanishing point (requires manual focal length)"),
            ('2VP', "2-Point", "Two vanishing points (assumes principal point at center)"),
            ('3VP', "3-Point", "Three vanishing points (derives principal point and camera shift)"),
        ],
        default='2VP',
        update=_on_solver_property_update,
    )

    # VP1 (X, red) - right side lines converging to the right, off-screen
    vp1_line1_start: FloatVectorProperty(
        name="VP1 Line1 Start", size=2, default=(0.45, 0.52),
    )
    vp1_line1_end: FloatVectorProperty(
        name="VP1 Line1 End", size=2, default=(0.85, 0.59),
    )
    vp1_line2_start: FloatVectorProperty(
        name="VP1 Line2 Start", size=2, default=(0.45, 0.80),
    )
    vp1_line2_end: FloatVectorProperty(
        name="VP1 Line2 End", size=2, default=(0.85, 0.74),
    )

    # VP2 (Y, green) - left side lines converging to the left, just off edge
    vp2_line1_start: FloatVectorProperty(
        name="VP2 Line1 Start", size=2, default=(0.15, 0.60),
    )
    vp2_line1_end: FloatVectorProperty(
        name="VP2 Line1 End", size=2, default=(0.55, 0.52),
    )
    vp2_line2_start: FloatVectorProperty(
        name="VP2 Line2 Start", size=2, default=(0.15, 0.73),
    )
    vp2_line2_end: FloatVectorProperty(
        name="VP2 Line2 End", size=2, default=(0.55, 0.81),
    )

    # VP3 (Z, blue) - near-vertical lines converging far above
    vp3_line1_start: FloatVectorProperty(
        name="VP3 Line1 Start", size=2, default=(0.40, 0.85),
    )
    vp3_line1_end: FloatVectorProperty(
        name="VP3 Line1 End", size=2, default=(0.43, 0.35),
    )
    vp3_line2_start: FloatVectorProperty(
        name="VP3 Line2 Start", size=2, default=(0.60, 0.85),
    )
    vp3_line2_end: FloatVectorProperty(
        name="VP3 Line2 End", size=2, default=(0.57, 0.35),
    )

    # Origin point - where the world origin projects onto the image
    origin_point: FloatVectorProperty(
        name="Origin Point", size=2, default=(0.50, 0.75),
    )

    # Horizon line (1VP mode) - defines camera roll / up direction
    horizon_start: FloatVectorProperty(
        name="Horizon Start", size=2, default=(0.2, 0.5),
    )
    horizon_end: FloatVectorProperty(
        name="Horizon End", size=2, default=(0.8, 0.5),
    )

    # Focal length for 1VP mode (user-provided)
    focal_length_1vp: FloatProperty(
        name="Focal Length (mm)",
        description="Camera focal length for 1-point perspective",
        default=50.0,
        min=1.0,
        soft_min=10.0,
        soft_max=300.0,
        update=_on_solver_property_update,
    )

    # Axis assignment
    vp1_axis: EnumProperty(
        name="VP1 Axis",
        description="World axis corresponding to vanishing point 1",
        items=[
            ('X', "X", "X axis"),
            ('Y', "Y", "Y axis"),
            ('Z', "Z", "Z axis"),
        ],
        default='X',
        update=_on_solver_property_update,
    )
    vp2_axis: EnumProperty(
        name="VP2 Axis",
        description="World axis corresponding to vanishing point 2",
        items=[
            ('X', "X", "X axis"),
            ('Y', "Y", "Y axis"),
            ('Z', "Z", "Z axis"),
        ],
        default='Y',
        update=_on_solver_property_update,
    )

    # Reference distance
    ref_distance_enabled: BoolProperty(
        name="Reference Distance",
        description="Enable reference distance for scale",
        default=False,
    )
    ref_distance: FloatProperty(
        name="Distance",
        description="Real-world distance between reference points",
        default=1.0,
        min=0.001,
        soft_min=0.01,
        soft_max=1000.0,
        unit='LENGTH',
        update=_on_ref_distance_update,
    )
    ref_point_a: FloatVectorProperty(
        name="Ref Point A", size=2, default=(0.40, 0.70),
    )
    ref_point_b: FloatVectorProperty(
        name="Ref Point B", size=2, default=(0.60, 0.70),
    )

    # Principal point (manual override, only used in 2VP mode)
    use_custom_pp: BoolProperty(
        name="Custom Principal Point",
        description="Override the assumed image-center principal point (optical center of the lens)",
        default=False,
        update=_on_solver_property_update,
    )
    principal_point: FloatVectorProperty(
        name="Principal Point", size=2, default=(0.5, 0.5),
        update=_on_solver_property_update,
    )

    # Background image opacity
    bg_alpha: FloatProperty(
        name="Image Opacity",
        description="Opacity of the camera background image",
        default=0.37,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_bg_alpha_update,
    )

    # Background image display depth
    bg_display_depth: EnumProperty(
        name="Image Depth",
        description="Display background image in front of or behind the 3D scene",
        items=[
            ('BACK', "Back", "Behind the 3D scene"),
            ('FRONT', "Front", "In front of the 3D scene"),
        ],
        default='BACK',
        update=_on_bg_depth_update,
    )

    # VP area fill toggle
    show_fill: BoolProperty(
        name="VP Fill",
        description="Show filled area between VP line pairs",
        default=True,
        update=_on_overlay_update,
    )

    # Camera selection
    target_camera: EnumProperty(
        name="Camera",
        description="Camera to configure",
        items=_get_camera_items,
    )


# All draggable control point property names
CONTROL_POINT_NAMES = [
    "vp1_line1_start", "vp1_line1_end",       # 0, 1
    "vp1_line2_start", "vp1_line2_end",        # 2, 3
    "vp2_line1_start", "vp2_line1_end",        # 4, 5
    "vp2_line2_start", "vp2_line2_end",        # 6, 7
    "vp3_line1_start", "vp3_line1_end",        # 8, 9
    "vp3_line2_start", "vp3_line2_end",        # 10, 11
    "origin_point",                             # 12
    "ref_point_a", "ref_point_b",              # 13, 14
    "principal_point",                          # 15
    "horizon_start", "horizon_end",            # 16, 17
]

# Default values for resetting (3/4 angle, horizon at bottom third)
CONTROL_POINT_DEFAULTS = {
    "vp1_line1_start": (0.45, 0.52),
    "vp1_line1_end": (0.85, 0.59),
    "vp1_line2_start": (0.45, 0.80),
    "vp1_line2_end": (0.85, 0.74),
    "vp2_line1_start": (0.15, 0.60),
    "vp2_line1_end": (0.55, 0.52),
    "vp2_line2_start": (0.15, 0.73),
    "vp2_line2_end": (0.55, 0.81),
    "vp3_line1_start": (0.40, 0.85),
    "vp3_line1_end": (0.43, 0.35),
    "vp3_line2_start": (0.60, 0.85),
    "vp3_line2_end": (0.57, 0.35),
    "origin_point": (0.50, 0.75),
    "ref_point_a": (0.40, 0.70),
    "ref_point_b": (0.60, 0.70),
    "principal_point": (0.5, 0.5),
    "horizon_start": (0.2, 0.5),
    "horizon_end": (0.8, 0.5),
}

# Names grouped by VP for drawing/interaction filtering
VP2_POINT_NAMES = ["vp2_line1_start", "vp2_line1_end", "vp2_line2_start", "vp2_line2_end"]
VP3_POINT_NAMES = ["vp3_line1_start", "vp3_line1_end", "vp3_line2_start", "vp3_line2_end"]
HORIZON_POINT_NAMES = ["horizon_start", "horizon_end"]


def register():
    bpy.utils.register_class(MatchCamProperties)
    bpy.types.Scene.matchcam = bpy.props.PointerProperty(type=MatchCamProperties)


def unregister():
    del bpy.types.Scene.matchcam
    bpy.utils.unregister_class(MatchCamProperties)
