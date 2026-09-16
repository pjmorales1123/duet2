# Duet 2 dinner assets

Original procedural models created September 11, 2026 with Codex assistance,
under the repository's [MIT license](../../../LICENSE). No downloaded dinnerware
meshes, image textures, or external asset licenses are needed.

The canonical parameterized builders are in [dinner.py](../../dinner.py).
The eight `.xml` files are standalone MuJoCo inspection scenes: seven free
tableware objects and a cabinet with a passive drawer. Each includes its own
materials, lighting and inspection floor. They compile without robot assets.
They are exported examples; the running application uses the Python builders.

Regenerate them from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\export_dinner_assets.py
```

`manifest.json` records dimensions, masses, candidate grasp sites and hashes.
Lengths use metres, mass uses kilograms, and angles use radians. The drawer
inspection scene has its support surface at 0.76 m; the loose-object inspection
scenes use a floor at zero. Body origins are at the bottom center of each prop.

Vessels have compound wall collisions and open interiors. The mug has a hollow
handle, the bottle has an open neck, and the fork has four tines. Vessel bases
are 14 mm thick to improve contact stability at the 5 ms timestep. The spoon
bowl is a solid ellipsoid approximation. Liquid flow, breakage, deformability,
and accurate glass optics are not modeled. Box-approximated inertias and small
lightweight dimensions are deliberate first-baseline simplifications.

The drawer is a limited, damped slide joint. The application initializes its
open/closed inspection state only when resetting; no drawer actuator is present.
