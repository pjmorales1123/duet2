# Talos: Learned Dinner-Table Robotics

Camera-based neural skills coordinate two SO-101 arms through six dinner-table tasks and bottle relays in MuJoCo, with Speechmatics voice input, OpenVINO inference and preserved physical evaluation.

## Project description

Talos turns supported spoken or typed instructions into dinner-table manipulation by two SO-101 arms in MuJoCo. Six learned skills place a bottle, plate and mug, open a passive drawer, and retrieve the fork and spoon. Speechmatics transcribes commands; a bounded language interpreter and classical RGB checks select actions. Neural motor trajectories use camera features and joint feedback. A separate simulator-state monitor stops or scores execution.

The original learned dinner system passed 8/10 frozen sequences. Table-supported bottle relays passed 5/5 frozen trials in each direction; a composed relay-plus-dinner workflow passed 8/10. All failures remain. Later stereo correction during late mug placement passed 20/20 narrow workflows, as did the original baseline; frozen corrective images passed 0/20. This establishes bounded visual feedback, not general workspace coverage.

A human “Set the table” voice command completed all six skills. The hosted live-mug CPU demo also passed. Six-skill baseline execution is recorded on a legacy Intel laptop with Intel CPU/iGPU inference and UHD rendering. Core Ultra Series 2/3 eligibility remains unresolved; the latest live-mug profile needs separate Intel verification.

Broader work includes seven loose items, full tabletop yaw and stable alternative orientations. No complete jointly randomized seven-item scene has passed. Calibrated RGB-D and an adapted CLIPort runtime work on the RTX 4070, but no shared Talos spatial policy has been trained. Arbitrary reachable positions, general visual recovery, unrestricted language, pouring and airborne handoffs remain unfinished.

Original Talos contributions are MIT licensed; third-party components retain their own licenses. Source, compact training inputs, model provenance, protocols and successful/failed physical outcomes are preserved.

## Additional information

To try the hosted demo, keep “Learned dinner · live mug vision”, “CPU · OpenVINO”, the standard bottle region and seed 42. Enter “Set the table” and select “Run fresh trial”. Each run creates a new physical scene; allow about five minutes for all six steps. The CPU path requires no shared-GPU allocation. The local app also provides voice and six simultaneous cameras.

The 4:49.5 video contains the actual microphone command, a complete local dinner sequence, a table-supported bottle relay and a hosted trial excerpt. Accelerated footage is labeled. The hosted excerpt shows work in progress; it does not claim to show that trial finishing.

The repository contains a clear setup guide, immutable model exports, compact training inputs and exact result summaries. Complete historical traces, unsuccessful experiments and earlier materials are preserved under the development-snapshot-2026-09-14 tag. Private recordings, credentials and editing files are excluded.

Core Ultra Series 2/3 eligibility remains unresolved. The included Intel evidence is explicitly for the legacy i7-10850H baseline; it does not establish qualifying Core Ultra execution or separate Intel verification of the latest live-mug profile. No complete jointly randomized seven-item scene or trained shared spatial policy is claimed.

## Submission settings

Online · Home Automation · Intel Online. Platform tags: Speechmatics api, HuggingFace Spaces, Nvidia. OpenVINO, MuJoCo and PyTorch are described in the project because the platform taxonomy does not list them.

- Source: https://github.com/jannissio/talos-ai-infra
- Demo: https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics
- [Current media and checks](README.md)

Final submission remains a user action.
