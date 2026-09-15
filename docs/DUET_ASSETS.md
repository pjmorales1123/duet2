# Duet 2 scene assets

## Evening service environment v1

Duet 2 replaces the generic laboratory presentation with an indigo/coral dinner
service environment. The first revision changes the visual world without
changing the graspable objects' collision dimensions, masses, target locations,
or grasp-site coordinates.

### Project-bound generated backdrop

File: simulation_lab/assets/duet/duet-evening-wall-v1.png

Created with the built-in image-generation workflow on 2026-09-15 for Duet 2.
It is an original abstract indigo dining-wall texture with coral and amber
architectural motifs. It has no text, logos, people, dinnerware, robots, or
brand identifiers.

Prompt summary: abstract evening dining-room wall texture; indigo plaster,
coral and warm-gold geometric arches; low visual noise; wide background;
no furniture, food, people, robots, readable text, logos or watermarks.

The texture is mounted as a non-colliding wall behind the table. The generated
image affects camera observations. New Duet training data must therefore be
collected with this scene; inherited camera-trained weights may not be claimed
to generalize to it without evaluation.

### Code-built visual assets

- Coral left work mat and cobalt right work mat.
- Amber service runner and target rings.
- Amber/coral centerpiece and backdrop sill.
- New materials and object labels for the Duet dinner service.

All are visual-only except for the inherited graspable dinnerware. They are
implemented in simulation_lab/dinner.py. The first revision intentionally
leaves physical geometry unchanged so controller compatibility can be measured
before a later geometry pass.

## Next asset pass

The first procedural form pass is complete: sunray dinner plates, a banded and
fluted cobalt cup, a plum carafe with amber bands, decorated cutlery, and a
coral/amber drawer facade. These are non-colliding overlays on the established
physical proxies.

Before changing collision envelopes or grasp sites, collect fresh Duet camera
demonstrations and measure the current controller across the declared training
split. A later geometry pass can then alter physical forms in small, measured
increments.
