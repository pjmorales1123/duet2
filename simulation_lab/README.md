# Simulation application

Use the [root quickstart](../README.md#run-locally) and [setup guide](../docs/SETUP.md). The source files in this folder implement the browser, physical scenes, programmed controllers, learned motion, camera observations, safety/scoring and optional voice.

`server.py` serves the local interface; `engine.py` coordinates scene state; `dinner.py` builds the dinner scene; `learned_dinner.py` runs the learned skill sequence; `mug_visual_profile.py` verifies the selected live-mug profile; `public_trial.py` supplies isolated hosted trials. `web/` contains the browser interface and `assets/` contains the attributed robot and procedural dinner geometry.

The [architecture](../docs/robotics/FINAL_ARCHITECTURE.md) explains the boundaries between learned control, programmed execution and independent scoring. Runtime source files bound by the promoted model profile must keep their exact bytes unless a new controller is evaluated and promoted.
