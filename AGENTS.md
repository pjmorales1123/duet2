# Duet 2 project instructions

Duet 2 modifies the supplied Talos baseline. Preserve upstream attribution, licenses, and immutable model/evidence artifacts. Existing upstream results are not Duet 2 measurements.

Read docs/DUET_IMPLEMENTATION_PLAN.md and docs/DUET_PROGRESS.md before work. Update the tracker with verification evidence after each unit; commit logical changes and preserve a rollback point before refactors.

Keep modules focused by responsibility. Inspect line counts and structural maps before reading large files. Preserve the working simulation_lab architecture while adding small scene, training, evaluation, and recovery modules. No broad reorganization during deadline work.

Use real grasp/contact physics. No teleportation or hidden teacher fallback in learned trials. Keep training/validation/evaluation separate. Display actual outcomes and retain failures.

All secrets stay in ignored .env. credentials.md is ignored and records configured services. No new accounts are configured. Fetch primary library documentation before API changes; Context7 and sequential-thinking were unavailable at kickoff. The /init command was unavailable, so these instructions were authored manually.

Validate UI changes in a browser and control changes through physical rollouts. Hardware and performance claims require actual logs. User reports organizer acceptance of 8th-generation Intel; report the exact machine accurately.
