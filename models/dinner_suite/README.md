# Learned dinner suite, revision v6

The measured submission baseline composes six learned skills: bottle, plate,
mug, drawer, fork and spoon. All six development scenes and exposed seed 42
passed. The frozen OpenVINO evaluation completed **8/10** full dinner scenes.
Seed 2026091902 failed mug placement (14.59 mm XY error); seed 2026091905 failed
spoon placement (13.61 mm). The unchanged acceptance threshold is 8 mm. All ten
reports, including failures, are in [the evidence](../../docs/robotics/evidence/learned-dinner-v6/summary.json).

This is a finite task-preset starting distribution with fixed destinations. It
does not establish arbitrary reachable-position coverage, arbitrary language,
continuous visual correction, pouring or airborne handoffs. The suite includes
the separately evaluated [bottle relay models](../bottle_relays/README.md).

Classical RGB geometry conditions a neural motor trajectory. Measured joint
feedback regulates progress. The five new models retain the previous motor
targets of the inactive arm to avoid an abrupt transition after parking.
A separate simulator-state monitor checks physical contacts, release, stable
placement and parked arms. It never generates learned motor actions. No teacher,
action retrieval, attachments, teleportation or hidden forces run in deployment.

Five new models were trained on the RTX 4070 using 201 eligible compact inputs.
All 201 aligned action replays passed before training. See the
[exact training inputs and commands](../../training/dinner_suite/README.md).
The preserved original bottle model and all previous evidence are unchanged.

Every packaged OpenVINO FP32 export passed numerical parity. The benchmark
files describe this AMD/NVIDIA PC, not final Intel hardware execution.

```powershell
./start-lab.ps1 -Learned -DinnerSuite models/dinner_suite/suite.json
./.venv-training/Scripts/python.exe scripts/evaluate_learned_dinner.py --suite models/dinner_suite/suite.json --skills bottle,plate,mug,drawer,fork,spoon --seed 2026091901 --output .run/dinner-v6-reproduction.json
```

Choose learned dinner control and say/type "set the table". The reproduction
seed above is exposed. Use a fresh output path and retain the 10 GiB reserve.
Original Talos models, data and code use MIT; upstream robot notices remain
separate. Further visual-feedback and coverage experiments must preserve this
baseline and disclose their own training/development/evaluation splits.
