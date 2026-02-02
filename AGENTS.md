# Agent Context: MatchCam

This document is for AI agents picking up development on this project. It covers architecture, conventions, gotchas, and the math model so you can make changes without breaking things.

## What This Is

A Blender 4.3+ addon that matches a 3D camera to a 2D background photograph using vanishing points. The user drags line handles over edges in the photo; the addon solves for focal length, rotation, and position in real-time. Think fSpy, but as a native Blender addon.

## Architecture Overview

```
properties.py  ──>  operators.py  ──>  solver.py
                         │
                    drawing.py
                         │
                      ui.py
```

- **properties.py**: All persistent state lives in `MatchCamProperties`, a `PropertyGroup` registered on `bpy.types.Scene.matchcam`. Control point positions are stored as `FloatVectorProperty(size=2)` in normalized (0-1) image coordinates. The `CONTROL_POINT_NAMES` list defines the canonical ordering of points (index matters for hit testing and undo).

- **operators.py**: The core is `MATCHCAM_OT_interact`, a persistent modal operator that runs while the addon is enabled. It handles mouse events (drag, hover, precision mode, axis constraint), maintains a custom undo/redo stack, and calls the solver on every control point change. Other operators: setup (camera + image), reset, enable toggle.

- **solver.py**: Pure math, no Blender imports. Takes normalized (0-1) control point positions + settings, returns a `SolverResult` dataclass with focal length, quaternion rotation, location, FOV, and shift values. Uses no external dependencies beyond Python's `math` module.

- **drawing.py**: GPU overlay rendering using Blender's `gpu` module. All primitives (lines, circles, diamonds) are anti-aliased using the `SMOOTH_COLOR` shader with alpha-feathered edges. The precision loupe samples directly from the background image via `gpu.texture.from_image()`.

- **ui.py**: A single N-panel (`VIEW_3D` sidebar, category "MatchCam") with mode toggle, axis assignment, settings, and status readout.

## Coordinate Systems

There are **three** coordinate systems in play. Getting these confused is the #1 source of bugs:

1. **Normalized (0-1)**: All control points are stored this way. `(0, 0)` = top-left of image, `(1, 1)` = bottom-right. Y increases downward (like screen coords, unlike Blender's convention).

2. **Screen pixels**: Blender region coordinates. Y increases upward. Conversion via `normalized_to_screen()` / `screen_to_normalized()` in `drawing.py`, which use the camera frame rectangle `(x, y, width, height)`.

3. **Image-plane**: Used internally by the solver. Origin at image center, Y up, longer dimension spans [-1, 1]. Conversion via `relative_to_image_plane()` / `image_plane_to_relative()` in `solver.py`.

The normalized-to-screen flip is: `screen_y = frame_y + (1.0 - norm_y) * frame_height`.

## Blender Camera Conventions

- Blender camera looks down **-Z local axis** with **+Y up**
- The solver builds a world-to-camera rotation matrix (`view_rot`), then derives camera-to-world as its transpose
- Camera location = `-transpose(view_rot) @ translation_cam`
- Quaternion is extracted from the camera-to-world rotation matrix
- Camera shift: `shift_x = -pp[0] / 2.0`, `shift_y = -pp[1] / 2.0` (Blender's shift direction is opposite to principal point offset)

## The Solver Math

Based on the Guillou et al. paper, closely following fSpy's implementation:

1. Convert normalized control points to image-plane coords
2. Intersect line pairs to get vanishing points Fu, Fv (and optionally Fw for 3VP)
3. In 3VP mode: principal point = orthocenter of triangle(Fu, Fv, Fw)
4. Focal length from orthogonality constraint: `f² = -d(Fv,Puv) * d(Fu,Puv) - d(P,Puv)²` where Puv is P projected onto line Fu-Fv
5. Rotation columns: `normalize(Fu - PP, -f)` and `normalize(Fv - PP, -f)`, third column is their cross product
6. Axis assignment: row matrix from world-axis unit vectors, multiplied as `cam_rot @ axis_mat`
7. Translation: unproject origin point through camera model

**Key invariant**: The solver returns `None` for any invalid configuration (parallel lines, negative f², coincident VPs, etc.). The operator and UI must handle `None` gracefully.

## The Modal Operator

`MATCHCAM_OT_interact` is the most complex piece. Key design decisions:

- **Always-on while enabled**: Runs as a persistent modal. Uses `PASS_THROUGH` for unhandled events so Blender UI (pan, zoom, menus, N-panel) keeps working normally.
- **Custom undo stack**: Blender's built-in undo doesn't work inside modals. We maintain `_undo_stack` and `_redo_stack` as lists of point snapshots (dicts). Ctrl+Z/Ctrl+Shift+Z are intercepted and return `RUNNING_MODAL` to prevent them from reaching Blender's undo system.
- **Precision mode (Shift+drag)**: Applies a 0.25x speed factor using delta-based positioning relative to `_last_mouse_screen`. Stores the handle screen position (not mouse position) in `scene["_matchcam_drag_screen"]` for the loupe.
- **Axis constraint (Ctrl+drag)**: Locks movement to horizontal or vertical based on dominant drag direction from `_drag_start_screen`.
- **Auto camera return**: If the user accidentally orbits out of camera view, the modal auto-switches back to camera view.
- **State on scene**: Hover index, precision mode flag, drag screen position, and drag index are stored as scene ID properties (`scene["_matchcam_*"]`) so the draw callback can read them.

## GPU Drawing

All rendering uses `SMOOTH_COLOR` shader with vertex-alpha feathering for anti-aliasing:
- Lines: drawn as quads with 1px transparent fringe on each side
- Circles: triangle fan with transparent outer ring
- Annulus (loupe border): 4 concentric vertex rings with alpha gradient
- Diamonds (VP indicators): inner fill + outer transparent fringe

The precision loupe uses `gpu.texture.from_image(image)` to sample the background image at full resolution, rendered as a UV-mapped triangle fan (circular clip) with the `IMAGE` builtin shader.

## Transient Scene Properties

These are stored via `scene["key"]` (Blender ID properties, not `bpy.props`):

| Key | Type | Set by | Read by |
|---|---|---|---|
| `_matchcam_hover_idx` | int | operator | drawing |
| `_matchcam_precision` | bool | operator | drawing |
| `_matchcam_drag_screen` | (float, float) | operator | drawing |
| `_matchcam_drag_idx` | int | operator | drawing |
| `_matchcam_valid` | bool | operator/_run_solver | ui |
| `_matchcam_focal_mm` | float | operator/_run_solver | ui |
| `_matchcam_hfov` | float | operator/_run_solver | ui |
| `_matchcam_vfov` | float | operator/_run_solver | ui |

## Control Point Index Map

The `CONTROL_POINT_NAMES` list in `properties.py` defines the canonical ordering. Index ranges:

| Indices | Group | Color |
|---|---|---|
| 0-3 | VP1 (line1 start/end, line2 start/end) | Red |
| 4-7 | VP2 | Green |
| 8-11 | VP3 | Blue |
| 12 | Origin point | Yellow |
| 13-14 | Reference points A, B | Yellow |

This ordering is used by hit testing, hover highlighting, undo snapshots, and loupe color selection.

## Common Pitfalls

1. **Attribute vs dict access**: Blender PropertyGroup values use attribute access (`props.vp1_line1_start`). Transient state uses dict access (`scene["_matchcam_hover_idx"]`). Mixing these up causes `AttributeError` or `KeyError`.

2. **FloatVectorProperty assignment**: Setting a `FloatVectorProperty` requires a tuple: `setattr(props, name, (x, y))`. Reading returns a Blender vector-like object; index with `[0]`, `[1]`.

3. **Shader bind ordering**: When switching between `SMOOTH_COLOR`, `UNIFORM_COLOR`, and `IMAGE` shaders in the same draw callback, you must call `.bind()` on the new shader before setting uniforms. The draw callback may switch shaders multiple times.

4. **Framebuffer vs image sampling**: Earlier versions read from the GPU framebuffer for the loupe, which picked up Blender's own UI overlays. Current code samples from the background image directly via `gpu.texture.from_image()`.

5. **Y-axis flip**: Normalized coords have Y=0 at top; screen coords have Y=0 at bottom; image-plane coords have Y=0 at center pointing up; image pixel coords (and UV coords) have Y=0 at bottom. Every conversion must account for this.

6. **Modal event return values**: `RUNNING_MODAL` consumes the event. `PASS_THROUGH` lets it propagate to Blender. Getting this wrong either breaks Blender's UI or prevents MatchCam from working. The undo handler must return `RUNNING_MODAL` even when there's nothing to undo, to prevent Blender's undo from firing.

7. **Solver returns None**: Any drawing or operator code that calls the solver must handle `None`. The solver rejects many configurations (parallel lines, negative focal length, degenerate triangles).

## Planned Work

From `notes.txt` and known issues:

- Real-time updates when the reference distance slider changes (currently requires a handle drag to trigger the solver)
- Exposed slider for background image opacity (default ~87%)
- Camera selection: if cameras already exist, let the user choose which to configure or create a new one; handle existing background images gracefully
- 1VP mode (future)

## Branch Conventions

Feature work is done on `feature/<name>` branches off `master`. Current branches:
- `master`: stable
- `feature/undo-ux`: merged (custom undo, precision mode, auto camera return, default VP positions)
- `feature/loupe-improvements`: current (AA rendering, image-based loupe, circular clip, axis-colored loupe)

## Testing

No automated tests. Testing is done manually in Blender:
1. Install addon (or reload via F3 > "Reload Scripts")
2. Enable MatchCam, load an image via Setup
3. Drag handles and verify camera updates
4. Test undo/redo, precision mode, axis constraint
5. Switch between 2VP and 3VP modes
6. Verify status readout in the N-panel
