# Experimental learned bottle model

This checkpoint generates 12 nominal joint targets from three initial RGB camera views and internal trajectory progress. A programmed motor-feedback guard pauses progress when the left arm lags. It is an initial-image-conditioned neural motion primitive, not a general VLA, continuous visual controller, or language model.

Training used eleven physically verified bottle demonstrations, including upright, sideways, small position offsets and a recovery start. Earlier evaluation obtained 9/11 familiar cases and 3/6 then-held-out cases. Those six cases are now exposed. The OpenVINO CPU export reproduced all five original familiar cases; this does not establish arbitrary-placement generalization.

A later, broader evaluation of ten new full-scene seeds (2026091201–2026091210) completed **0/10** tasks. Failures included missed grasps, loss of finger support and timeout. The default scene and familiar sideways case still pass. This checkpoint is a working local baseline, not a robust final submission model; broader demonstrations are being collected separately.

`primitive.safetensors` contains the PyTorch weights. `primitive.xml`/`.bin` contain FP32 OpenVINO network weights; `openvino.json` selects CPU execution. `visual.npz` stores camera PCA parameters, not demonstration actions. Normalization and model metadata are JSON. No pickle model loading is needed.

The live application uses RGB and motor state for action generation. A separate simulator-state observer checks grasp support, placement, disturbances and timeout, and may stop execution. It never provides action targets or teacher corrections. Cameras render independently of the browser display. Inference is synchronous with physics and can reduce wall-time speed.

Only the bottle and its trained destination are supported by this model. Programmed dinner skills are a separate, labeled mode. No handoff, arbitrary destination, liquid simulation, or real robot transfer claim is made.

Model weights and project code: MIT license. SO-101 scene asset provenance and its separate Apache-2.0 license are documented in the simulator assets directory.
