# mug-visual-correction-v2: complete physical comparison

This separately declared comparison changes renderer setup order only. The V1 mug controller, observer, motor, timing and physical criteria remain unchanged. All failures and denominators are retained.

| Mode | Standard workflows | Left-reach workflows | Standard mugs | Left-reach mugs |
| --- | --- | --- | --- | --- |
| baseline | 10/10 | 10/10 | 10/10 | 10/10 |
| live | 10/10 | 10/10 | 10/10 | 10/10 |
| frozen | 0/10 | 0/10 | 0/10 | 0/10 |

The latest completed split is **evaluation**: 60 attempted workflows, 262,414 retained state frames, 504 visual correction queries. Its gate **passes**. Failed checks: none.

The [protocol](protocol.json), [development results](development-audit.json), every scene/report/state trace, exact frozen runtime sources and console provenance are retained. Original XML files are stored beside the portable scenes. Package verification recounts outcomes and compares prefixes before the first correction, preserving mismatches as a failed criterion. Earlier missing-test-runner evidence is kept where applicable; no test ran during that failed launcher.

Models remain in [the stereo observer package](../../../../models/mug_keypoint_observer_v1/README.md) and [the limited motor package](../../../../models/mug_correction_motor_v2/README.md). Their manifests and the selected dinner suite are frozen in development-freeze.json. This experiment creates no replacement weights. The selected dinner/browser/hosted baseline remains unchanged pending a separate promotion decision.

Run the package-only integrity and paired-result check from the repository:

```powershell
.\.venv-training\Scripts\python scripts/package_mug_visual_correction.py --protocol docs/robotics/experiments/mug-visual-correction-v2.json --verify-only
```

No broad workspace, arbitrary-language, Intel-hardware or human-voice claim follows from this finite comparison. GitHub and Hugging Face remain private until the final release. Final Submit is exclusively the user's action.
