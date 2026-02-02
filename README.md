# MatchCam

A Blender addon that matches a camera's perspective to a background image using vanishing points. Works like [fSpy](https://fspy.io/) but runs entirely inside Blender with real-time feedback.

## Requirements

- Blender 4.3+

## Installation

1. Download or clone this repository
2. In Blender: Edit > Preferences > Add-ons > Install
3. Select the `matchcam` folder (or zip it first)
4. Enable "MatchCam" in the addon list

## Usage

1. Open the **MatchCam** tab in the 3D Viewport sidebar (N-panel)
2. Click **Setup Camera + Load Image** to create a camera and load your reference photo
3. Click **Enable** to start the interactive overlay
4. Drag the red and green line handles to align them with parallel edges in your photo:
   - **Red lines (VP1)**: align to one set of parallel edges (default: X axis)
   - **Green lines (VP2)**: align to a different set of parallel edges (default: Y axis)
5. The camera's focal length, rotation, and position update in real-time

### Modes

- **2-Point**: Two vanishing points. Assumes the principal point is at the image center (or a custom location). Suitable for most architectural shots.
- **3-Point**: Three vanishing points. Blue lines define a third VP (default: Z axis). The principal point and camera shift are derived automatically from the triangle orthocenter of all three VPs.

### Controls

| Action | Effect |
|---|---|
| Left-click + drag | Move a control point handle |
| Shift + drag | Precision mode (1/4 speed) with magnified loupe |
| Ctrl + drag | Constrain to horizontal or vertical axis |
| Right-click / Esc (during drag) | Cancel drag, restore previous position |
| Ctrl+Z | Undo last handle move |
| Ctrl+Shift+Z | Redo |

### Panel Options

- **Mode**: Switch between 2-Point and 3-Point perspective
- **Axis Assignment**: Choose which world axis each VP corresponds to (X, Y, or Z)
- **Custom Principal Point** (2VP only): Override the default image-center principal point
- **Reference Distance**: Place two reference points and enter a known real-world distance between them to set scene scale
- **Reset Control Points**: Return all handles to default positions

## How It Works

The solver uses the classic vanishing point calibration method:

1. Two pairs of user-drawn lines intersect to give two vanishing points (Fu, Fv)
2. The orthogonality constraint between VP directions yields the focal length
3. A rotation matrix is built from the normalized VP direction vectors
4. An axis assignment matrix maps VP directions to world axes
5. The origin control point defines where the world origin projects onto the image, which determines camera translation

Reference: "Using Vanishing Points for Camera Calibration and Coarse 3D Reconstruction from a Single Image" (Guillou, Meneveaux, Maisel, Bouatouch). Implementation closely follows fSpy's solver.

## File Structure

```
matchcam/
  __init__.py      # bl_info, register/unregister
  properties.py    # PropertyGroup: control points, settings, defaults
  operators.py     # Persistent modal (drag interaction, undo), setup, reset, enable
  solver.py        # VP computation, focal length, rotation, translation math
  drawing.py       # GPU overlay: AA lines, handles, VP indicators, precision loupe
  ui.py            # N-panel (sidebar UI)
```

## Known Limitations

- Distance slider does not update the camera in real-time (planned)
- No background image opacity control in the panel (planned)
- Setup always creates a new camera; no option to choose an existing one (planned)
