"""
MatchCam - Match camera perspective to background image using vanishing points.

A Blender addon that replicates fSpy-like camera matching entirely within Blender.
Draw two pairs of convergence lines on a background image, and the camera's focal
length, rotation, and position update in real-time to match the perspective.
"""

bl_info = {
    "name": "MatchCam",
    "author": "",
    "version": (0, 1, 0),
    "blender": (4, 3, 0),
    "category": "Camera",
    "location": "View3D > Sidebar > MatchCam",
    "description": "Match camera perspective to background image using vanishing points",
    "doc_url": "",
    "tracker_url": "",
}

from . import properties
from . import operators
from . import ui


def register():
    properties.register()
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
    properties.unregister()
