# Demonstrated architecture

Speechmatics transcribes the local microphone input. A limited command grammar maps supported instructions to a task plan; classical RGB observations choose supported object/arm actions. The typed path uses the same planner.

For the six dinner skills, a trained trajectory network maps initial image features and trajectory time to twelve joint targets. Joint feedback regulates progress. The bottle relay consists of two independently learned legs, with real contact, release onto the shared tabletop and a second grasp. Both directions have separately measured outcomes.

The selected `models/dinner_visual_mug_v1/profile.json` adds a trained stereo image-point observer and calibrated triangulation during late mug placement. A small corrective motor network maps the observed offset and current joints to bounded corrections. This feedback is queried repeatedly within that portion of the motion. The original trajectory controller remains the baseline for the other skills.

An independent simulator-state monitor can stop and score learned execution. Learned controllers apply motor targets; they cannot teleport objects, attach them to hands or silently fall back to the programmed teacher. Programmed execution is a separate selectable mode and uses exact scene state.

MuJoCo performs the physics. The local FastAPI browser exposes six live camera views, manual joint/object controls, speech, progress, cancellation and failure reports. The Gradio Space exposes a smaller isolated trial with one selected camera. CPU inference uses the preserved OpenVINO exports; optional shared-GPU inference is a separately requested hosting path.

The promoted profile hashes its runtime dependencies, model exports, comparison protocol and gate reports. Cleanup retains those bytes and paths exactly. Version suffixes in these immutable model directories identify measured artifacts; they are not interchangeable drafts.

See [results and limitations](../EVIDENCE.md), [setup](../SETUP.md) and the [model index](../../models/INDEX.md). The complete historical architecture notes and unsuccessful broader-policy work remain in the development archive. No complete randomized seven-item controller or unrestricted language policy is claimed.
