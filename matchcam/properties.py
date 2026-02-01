import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
)


class MatchCamProperties(bpy.types.PropertyGroup):
    """Properties for MatchCam camera matching."""

    enabled: BoolProperty(
        name="Enable",
        description="Enable MatchCam overlay and interaction",
        default=False,
    )

    # Mode: 2VP or 3VP
    mode: EnumProperty(
        name="Mode",
        description="Number of vanishing points to use",
        items=[
            ('2VP', "2-Point", "Two vanishing points (assumes principal point at center)"),
            ('3VP', "3-Point", "Three vanishing points (derives principal point and camera shift)"),
        ],
        default='2VP',
    )

    # VP1 - first pair of convergence lines
    vp1_line1_start: FloatVectorProperty(
        name="VP1 Line1 Start", size=2, default=(0.3, 0.35),
    )
    vp1_line1_end: FloatVectorProperty(
        name="VP1 Line1 End", size=2, default=(0.7, 0.4),
    )
    vp1_line2_start: FloatVectorProperty(
        name="VP1 Line2 Start", size=2, default=(0.3, 0.65),
    )
    vp1_line2_end: FloatVectorProperty(
        name="VP1 Line2 End", size=2, default=(0.7, 0.6),
    )

    # VP2 - second pair of convergence lines
    vp2_line1_start: FloatVectorProperty(
        name="VP2 Line1 Start", size=2, default=(0.35, 0.3),
    )
    vp2_line1_end: FloatVectorProperty(
        name="VP2 Line1 End", size=2, default=(0.4, 0.7),
    )
    vp2_line2_start: FloatVectorProperty(
        name="VP2 Line2 Start", size=2, default=(0.65, 0.3),
    )
    vp2_line2_end: FloatVectorProperty(
        name="VP2 Line2 End", size=2, default=(0.6, 0.7),
    )

    # VP3 - third pair of convergence lines (for 3VP mode)
    vp3_line1_start: FloatVectorProperty(
        name="VP3 Line1 Start", size=2, default=(0.34, 0.61),
    )
    vp3_line1_end: FloatVectorProperty(
        name="VP3 Line1 End", size=2, default=(0.31, 0.31),
    )
    vp3_line2_start: FloatVectorProperty(
        name="VP3 Line2 Start", size=2, default=(0.68, 0.61),
    )
    vp3_line2_end: FloatVectorProperty(
        name="VP3 Line2 End", size=2, default=(0.71, 0.35),
    )

    # Origin point - where the world origin projects onto the image
    origin_point: FloatVectorProperty(
        name="Origin Point", size=2, default=(0.5, 0.5),
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
    )
    ref_point_a: FloatVectorProperty(
        name="Ref Point A", size=2, default=(0.4, 0.5),
    )
    ref_point_b: FloatVectorProperty(
        name="Ref Point B", size=2, default=(0.6, 0.5),
    )

    # Principal point (manual override, only used in 2VP mode)
    use_custom_pp: BoolProperty(
        name="Custom Principal Point",
        description="Override default image-center principal point",
        default=False,
    )
    principal_point: FloatVectorProperty(
        name="Principal Point", size=2, default=(0.5, 0.5),
    )


# All draggable control point property names
CONTROL_POINT_NAMES = [
    "vp1_line1_start", "vp1_line1_end",
    "vp1_line2_start", "vp1_line2_end",
    "vp2_line1_start", "vp2_line1_end",
    "vp2_line2_start", "vp2_line2_end",
    "vp3_line1_start", "vp3_line1_end",
    "vp3_line2_start", "vp3_line2_end",
    "origin_point",
    "ref_point_a", "ref_point_b",
]

# Default values for resetting
CONTROL_POINT_DEFAULTS = {
    "vp1_line1_start": (0.3, 0.35),
    "vp1_line1_end": (0.7, 0.4),
    "vp1_line2_start": (0.3, 0.65),
    "vp1_line2_end": (0.7, 0.6),
    "vp2_line1_start": (0.35, 0.3),
    "vp2_line1_end": (0.4, 0.7),
    "vp2_line2_start": (0.65, 0.3),
    "vp2_line2_end": (0.6, 0.7),
    "vp3_line1_start": (0.34, 0.61),
    "vp3_line1_end": (0.31, 0.31),
    "vp3_line2_start": (0.68, 0.61),
    "vp3_line2_end": (0.71, 0.35),
    "origin_point": (0.5, 0.5),
    "ref_point_a": (0.4, 0.5),
    "ref_point_b": (0.6, 0.5),
}

# Names grouped by VP for drawing/interaction filtering
VP3_POINT_NAMES = ["vp3_line1_start", "vp3_line1_end", "vp3_line2_start", "vp3_line2_end"]


def register():
    bpy.utils.register_class(MatchCamProperties)
    bpy.types.Scene.matchcam = bpy.props.PointerProperty(type=MatchCamProperties)


def unregister():
    del bpy.types.Scene.matchcam
    bpy.utils.unregister_class(MatchCamProperties)
