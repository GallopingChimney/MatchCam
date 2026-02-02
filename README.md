# MatchCam

A Blender addon for matching a 3D camera's perspective to a 2D background photograph using vanishing points. Think [fSpy](https://fspy.io/), but running natively inside Blender with real-time feedback.

## Features

- **1-Point, 2-Point, and 3-Point perspective** matching modes
- **Real-time camera calibration** -- focal length, rotation, and position update as you drag
- **Interactive overlay** with color-coded draggable control point handles
- **Precision mode** (Shift+drag) with magnified loupe at 1/4 speed
- **Axis constraint** (Ctrl+drag) for horizontal/vertical alignment
- **Undo/Redo** (Ctrl+Z / Ctrl+Shift+Z) within the modal operator
- **Reference distance** scaling for real-world scene calibration
- **Custom principal point** override (2VP mode)
- **Configurable axis assignment** for each vanishing point
- **GPU-accelerated** anti-aliased overlay rendering

## Requirements

- Blender 4.3+

## Installation

1. Download the `matchcam` folder (or a release zip)
2. In Blender: **Edit > Preferences > Add-ons > Install**
3. Select the `matchcam` folder or zip file
4. Enable **MatchCam** in the addon list

## Usage

1. Open the **MatchCam** tab in the 3D Viewport sidebar (N-panel)
2. Click **Setup Camera + Load Image** to create a camera and load your reference photo
3. Click **Enable** to start the interactive overlay
4. Drag the line handles to align them with parallel edges in your photo:
   - **Red lines (VP1)**: one set of parallel edges (default: X axis)
   - **Green lines (VP2)**: another set of parallel edges (default: Y axis)
   - **Blue lines (VP3)**: third set (3-Point mode only, default: Z axis)
5. The camera updates in real-time as you adjust

### Perspective Modes

| Mode | Description |
|------|-------------|
| **1-Point** | Single vanishing point with manual focal length control |
| **2-Point** | Two vanishing points. Principal point at image center (or custom). Best for most architectural shots |
| **3-Point** | Three vanishing points. Principal point and camera shift derived automatically from the triangle orthocenter |

### Controls

| Action | Effect |
|--------|--------|
| Left-click + drag | Move a control point handle |
| Shift + drag | Precision mode (1/4 speed) with magnified loupe |
| Ctrl + drag | Constrain to horizontal or vertical axis |
| Alt + drag (VP diamond) | Pivot mode -- drag the vanishing point directly |
| Right-click / Esc (during drag) | Cancel drag, restore previous position |
| Ctrl+Z | Undo last handle move |
| Ctrl+Shift+Z | Redo |

### Panel Options

- **Mode** -- Switch between 1-Point, 2-Point, and 3-Point perspective
- **Axis Assignment** -- Choose which world axis each VP corresponds to (X, Y, or Z)
- **Focal Length** (1VP mode) -- Manual focal length slider
- **Custom Principal Point** (2VP only) -- Override the default image-center principal point
- **Reference Distance** -- Place two reference points and enter a known real-world distance to set scene scale
- **Reset Control Points** -- Return all handles to default positions
- **Lock / Keyframe Camera** -- Lock the solved camera or insert keyframes

## How It Works

The solver uses the classic vanishing point calibration method from the computer vision literature:

1. User-drawn line pairs intersect to define vanishing points
2. The orthogonality constraint between VP directions yields the focal length
3. A rotation matrix is constructed from normalized VP direction vectors
4. An axis assignment matrix maps VP directions to world axes
5. The origin control point defines where the world origin projects onto the image, determining camera translation

Based on the method described in: *"Using Vanishing Points for Camera Calibration and Coarse 3D Reconstruction from a Single Image"* (Guillou, Meneveaux, Maisel, Bouatouch).

## File Structure

```
matchcam/
  __init__.py      # Blender addon registration
  properties.py    # Scene properties: control points, settings, defaults
  operators.py     # Modal interaction, setup, reset, enable operators
  solver.py        # Camera calibration math (pure Python, no Blender imports)
  drawing.py       # GPU overlay: AA lines, handles, VP indicators, loupe
  ui.py            # N-panel sidebar UI
```


## License

Copyright (c) 2025 Felix. All rights reserved.

This software is proprietary. Redistribution, resale, and modification for redistribution are prohibited without express written permission from the copyright holder. See [LICENSE](LICENSE) for full terms.
