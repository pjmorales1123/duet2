# Cabinet-source spawn grasp design

## Purpose

Replace the passive cutlery drawer with one open, table-height cabinet zone. The zone groups the source objects visually without introducing a new raised contact surface or a drawer-opening dependency. The existing table remains the physical support surface.

## Source layout

The user-supplied dinner layout is the canonical FINAL arrangement, copied into versioned config/dinner-layout.json. cabinet_source.py independently chooses accessible source slots and applies seeded position/yaw variation. Glass and side plate remain fixed at final positions. Target Z is normalized to physical table support; the old elevated cutlery Z came from drawer supports and cannot remain floating. Explicit editor overrides still override initial poses only.

The cabinet zone is a non-blocking, visual open-front tray boundary sized to enclose the declared source-object footprint. It has no collision geometry and never moves objects. The real table continues to provide all support contacts.

## Spawn-aware grasping

Each grabbable object retains a local grasp frame in its object specification. At planning time, the controller reads the live free-body transform from MuJoCo and transforms that local frame into world space. Cutlery gripper-axis constraints therefore rotate with the fork or spoon's actual spawn yaw instead of using a fixed world-frame vector.

The existing arm-selection loop and IK/carry-path collision checks remain authoritative. A candidate must be reachable, collision-free, physically grasped by both fingers, and held unsupported before placement. This is an exact-state simulator teacher, not a camera policy; that scope will be displayed and retained in trial metadata.

## Task contract

Remove the drawer task, drawer-open prerequisite, and drawer item from the normal dinner workflow. The new sequence is:

`bottle -> plate -> mug -> fork -> spoon`

Fork and spoon are tabletop/cabinet-zone pickups. Their placement targets and physical release checks remain separate from their seed-defined source poses. Existing recordings and results that include drawer manipulation remain historic baseline artifacts and are not relabelled as results from the new workflow.

## Modules and interfaces

- `config/dinner-layout.json`: versioned approved final destinations.
- `simulation_lab/cabinet_source.py`: independent seeded, accessible starting poses.
- `simulation_lab/dinner.py`: cabinet-zone visual fixture and use of the canonical layout at scene creation.
- `simulation_lab/spawn_grasp.py`: transform local grasp constraints through a live object rotation.
- `simulation_lab/dinner_autonomy.py`: spawn-aware cutlery orientation, five-skill sequencing, and removal of drawer assumptions.
- `simulation_lab/command_task.py`, server/UI labels, and evaluation scripts: expose the five-skill contract only.
- Focused tests: layout loading, rotated grasp-axis transformation, no drawer prerequisite, and physical five-skill rollouts on selected seeds.

## Error handling and evidence

Invalid or incomplete canonical layout files fail scene construction with an actionable validation error. An unreachable rotated grasp remains a recorded planning failure; no fallback may write object state, apply forces, or attach the object. Verification will retain trial reports for both successful and failed seeds, and the progress tracker will distinguish the new five-skill workflow from prior drawer-based evidence.
