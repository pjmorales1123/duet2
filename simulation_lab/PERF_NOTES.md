# Live SmolVLA: correctness fixes + performance notes

Checkpoint-agnostic: the `lerobot`-version-drift fixes below apply to
whatever checkpoint lands in `models/duet-smolvla-v1` (a newer, longer-trained
checkpoint is expected here later), since they patch how `_load_policy()`
constructs the policy object, not anything about a specific checkpoint's
weights. Rerun `scripts/benchmark_live_vla.py` once a new checkpoint is in
place, ideally on the target demo hardware — the render-cost bottleneck below
is scene-complexity-bound rather than model-size-bound, so it should show up
on any machine, just less severely on a faster one.

Measured 2026-09-16 on a CPU-only dev machine (no GPU), Python 3.11.9,
`lerobot==0.3.2`, `transformers==5.12.1`, `torch==2.8.0+cpu`, against the
`models/duet-smolvla-v1` checkpoint. Reproduce with:

```powershell
python scripts/benchmark_live_vla.py --instruction "place the mug on the table"
```

## Where the live VLA is exercised

- **Nowhere in `tests/`** — no test imports `simulation_lab.vla_task` or
  exercises the `mode: "learned_vla"` language command. `scripts/verify_live_policy.py`
  looks similar but drives the older single-skill `LearnedBottleTask`
  (`learned_task.py`), not SmolVLA.
- **Manually**, via `python -m simulation_lab.server` → the engineering
  console or `/demo` dashboard's "Live VLA demo" tab → a free-text instruction
  sent as a `language` command with `mode: "learned_vla"` →
  `LabEngine._language_command` → `simulation_lab/vla_task.py`'s `SmolVLATask`.
- **Now also**: `scripts/benchmark_live_vla.py` (new) — runs the same
  `SmolVLATask` code path headless against a real dinner scene and reports
  load time, render/inference latency, and a real-time-factor verdict. This
  is the script to rerun before/after any future change here, and the one to
  point a live judge-facing test at if you want a pre-flight check that
  doesn't require the browser.
- The live VLA also has **no physical success scoring**: `SmolVLATask.update()`
  unconditionally reports `"succeeded"` once `duration_s` elapses, unlike the
  scripted teacher (`dinner_autonomy.py`), which is checked by
  `dinner_monitor.py`'s privileged scoring. A "did it actually work" signal
  for the live VLA would have to come from watching the camera feed / demo
  video, not from `task.status`.

## Correctness fixes (needed just to load the checkpoint at all)

The installed `lerobot==0.3.2` (PyPI) is not the same lerobot revision the
checkpoint was exported from (evidenced by `config.json` fields it no longer
declares, and a normalization architecture it no longer uses the same way).
All fixed in `simulation_lab/vla_task.py::_load_policy`:

1. **`config.json` has fields `SmolVLAConfig` doesn't accept** (`use_peft`,
   `pretrained_path`, `pretrained_revision`, `rtc_config`, `compile_model`,
   `compile_mode`) → filtered to `SmolVLAConfig`'s actual dataclass fields
   before construction.
2. **`normalization_mapping` values are plain strings, not `NormalizationMode`
   enum members** (draccus round-trips this transparently; a hand-built
   dataclass doesn't) → now forced to `NormalizationMode.IDENTITY` for every
   modality (see point 3, this replaces the parse-from-JSON approach).
3. **Installed `lerobot`'s `SmolVLAPolicy` builds real `Normalize`/`Unnormalize`
   buffers into the model itself** when a modality maps to `MEAN_STD`, and
   asserts they're non-infinite on every forward pass. This checkpoint's
   export-time lerobot normalized/unnormalized *outside* the model (a
   separate preprocessor/postprocessor pipeline — see
   `policy_preprocessor.json` / `policy_postprocessor.json` next to the
   checkpoint), which is exactly what `SmolVLATask.update()` already does by
   hand against `policy_preprocessor_step_5_normalizer_processor.safetensors`.
   Forcing `IDENTITY` skips creating those buffers, so the checkpoint (which
   was never saved with them) loads cleanly under `strict=True`, and the
   hand-rolled normalize/unnormalize in `update()` stays the single source of
   truth — unchanged.
4. **`transformers` now requires `accelerate`** for the `device_map="auto"`
   path `SmolVLMWithExpertModel` uses internally → added `accelerate` to
   `requirements.txt`.

None of these change model behavior — they only get the exact same math
running under a newer `lerobot`/`transformers` than the checkpoint was
exported with.

## Performance: what's actually slow

> **CORRECTED — read this first.** Two conclusions in the section below were
> wrong, and are kept only so the reasoning trail is visible:
>
> 1. **"Inference is ~21-25ms."** That was timing `select_action()`'s
>    *queue-pop* path. SmolVLA is an action-chunk policy (`chunk_size=50`): it
>    runs a real forward pass only when its 50-action queue empties. A real
>    forward pass on this machine is **~62.5s** (54.9 / 60.7 / 62.5 / 63.2s
>    across runs); a pop is **12ms**. The "one unexplained 48-62s outlier"
>    documented further down was never an outlier — it was the only real
>    inference in the run.
> 2. **"Rendering is the dominant cost."** It is ~220ms/frame against a 62.5s
>    forward pass — about 0.7% of a query that actually infers. Rendering was
>    still being done wastefully (49 of 50 queries rendered frames nothing
>    read; fixed, see `_needs_fresh_frames`), but it was never the bottleneck.
>
> Where the 62.5s goes: `num_steps=1` → 56.2s vs `num_steps=5` → 62.8s, so the
> denoising loop is ~1.6s/step (~10%). The other **~88% is VLM vision encoding**
> of two 512x512 images through SmolVLM2-500M on 4 CPU cores. That is the only
> place with real headroom, and it is what an OpenVINO port should target.
>
> **This has since been fixed.** The forward pass is now **5.25s** and the
> real-time factor **0.49** (was 0.05). The decisive cause was not algorithmic:
> the checkpoint ships **bfloat16** and this CPU has no bf16 instructions, so
> torch was *emulating* every op at an 8.9x cost. `policy.float()` alone is 8.8x;
> putting the vision tower on OpenVINO CPU FP32 takes it the rest of the way.
> Both are in `vla_openvino.py`. Every number in the section below is a bf16
> measurement and should be read as historical.
>
> Judge-facing summary of all of this: [`OPTIMIZATIONS.md`](../OPTIMIZATIONS.md).

**(superseded) The bottleneck is camera rendering, not model inference.**

| Stage | Cost | Budget | Verdict |
|---|---|---|---|
| Cold checkpoint load (first `learned_vla` command per server process) | ~55-70s | one-time | slow but one-time |
| Camera render (`mujoco.Renderer.render()`, per frame) | ~290-340ms median | — | **dominant cost** |
| Model inference (`policy.select_action`, 5 denoising steps) | ~21-25ms median (p95 ~26ms) | — | fine |
| Per query (2 frames + inference, every 20 ticks / 100ms sim time) | ~600-670ms | 100ms | **~6x over budget** |
| Resulting real-time factor while the VLA is driving | ~0.04-0.09 | ~1.0 | visibly stutters |

The physics loop (`engine.py::LabEngine._run`) batches at most 0.1s of sim
time per outer iteration, and `SmolVLATask.QUERY_EVERY_TICKS=20` ticks at a
0.005s timestep lands one inference call inside that same 0.1s window — so
"real-time" needs `2*render_ms + inference_ms < 100ms`. Right now it's
~600-670ms, roughly 6x over. In practice: every ~100ms of simulated motion,
the whole engine (physics *and* the HTTP server, since both run on the same
thread — see `engine.py`) stalls for ~600ms while a VLA query resolves.

### Diagnosed, not (yet) fixed: rendering cost is not resolution-bound

Tested 320x240, 160x120, and 80x60 renders of the same dinner scene: all land
in the same ~200-290ms band. Cutting render resolution buys nothing, so it's
not a fill-rate/pixel problem — it's per-frame CPU-side scene traversal
(`update_scene`) and/or vertex processing cost from scene complexity (two
SO-101 arms + full dinner geometry), not something a resolution change can
fix. Disabling shadows (already done, per-frame, in `_frame()`) measured a
genuine ~45% improvement (~245ms vs ~440ms with shadows on) — already
shipped, no further change made there.

**Not touched further:** additional render-flag changes (disabling
reflections/skybox/fog, changing image resolution) were considered and
rejected — the checkpoint was trained on frames rendered a specific way, and
changing that at inference time risks a train/inference visual mismatch that
would hurt actual task success for a real speed win of unclear size. That's
a worse trade for a live demo than the current stutter.

### Tried and reverted: skip loading pretrained VLM weights

`SmolVLMWithExpertModel.__init__` loads real
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` pretrained weights just to build
the architecture — `policy.load_state_dict(..., strict=True)` right after
overwrites 100% of them from this checkpoint anyway, so the pretrained
weights looked like pure waste (download + CPU copy). Tried
`cfg["load_vlm_weights"] = False` (builds the same architecture from config
only, random-initialized, matching the checkpoint's own field once
`load_state_dict` overwrites it).

**Measured slower, not faster**: ~147s cold start vs ~55-70s with weights
loaded. Root cause: the `load_vlm_weights=True` path passes
`low_cpu_mem_usage=True` to `from_pretrained`, which skips real parameter
initialization and streams checkpoint bytes straight into place; the
`from_config` fallback fully initializes ~500M random parameters first (real
`nn.init` calls), which apparently costs more than the download + copy it
was meant to avoid on this machine. **Reverted** — left as
`load_vlm_weights=True` (the checkpoint's own declared default). Worth
retrying only if someone finds a way to build the architecture on the `meta`
device (no real init, no real weights) before `load_state_dict`.

### (RESOLVED) Unexplained: one very slow first inference call

**Explained:** this was not an outlier or a warm-up artifact. It was the single
real forward pass in the run; every other sample was a 12ms queue pop. Forcing
repeated chunk refills gives 60.7 / 62.5 / 63.2s — steady, not first-call-only.
The original (wrong) reasoning is kept below.


Across every run, `inference_ms` has one massive outlier (48s-62s) among an
otherwise tight ~20-27ms distribution (median/p95 barely move). Always the
first `select_action()` call after `_load_policy()` returns. Not yet
root-caused — candidates not yet ruled out: lazy `torch`/MKL thread-pool
initialization, a lazy weight materialization deferred until first use, or
memory-pressure paging right after loading a large model. Steady-state
inference is fine either way (~20ms), so this only matters for the very
first VLA action after a fresh checkpoint load — worth a closer look before
a live demo where that first action is likely to be on camera.

## OpenVINO: measured, and what it can/can't buy here

This laptop is an **i5-8265U + Intel UHD 620**, and OpenVINO sees both:
`Core().available_devices == ['CPU', 'GPU']`. The repo already has a working OV
pattern to copy (`simulation_lab/bottle_refinement_runtime.py`,
`scripts/export_mug_keypoints.py`), and `openvino==2026.3.0` is already in
`requirements.txt` — so this is not a new dependency, just an unused one.

Benchmarked a conv-heavy toy net (4 convs, 1x3x256x256) three ways:

| Runtime | Latency | vs torch |
|---|---|---|
| torch CPU | 135.9 ms | 1.0x |
| OV CPU | 71.5 ms | **1.9x faster** (compile 0.2s) |
| OV GPU (UHD 620) | 48.7 ms | **2.8x faster** (compile 4.8s) |

> **(superseded)** The paragraph that stood here argued OV could not fix the lag
> — "inference is only ~20ms of the ~600ms query… real-time factor moves from
> ~0.05 to ~0.05. The Amdahl ceiling here is brutal." That reasoning was sound
> but built on the wrong premise: the ~20ms figure was the queue-pop path (see
> the correction at the top of this file), so the Amdahl fraction was inverted.
> Inference was ~99% of a real query, not 3% of it. **OV is now shipped on the
> VLA path and the real-time factor moved 0.05 → 0.49.** Kept for the trail.

### What was actually done

Measured on the real SmolVLA vision tower, not a toy net — one 512x512 image:

| Runtime | Latency | vs shipped |
|---|---|---|
| torch **bfloat16** (as the checkpoint ships) | 16906.7 ms | 1.0x |
| torch fp32 | 1904.4 ms | 8.9x |
| **OV CPU fp32** | **1605.6 ms** | **10.5x** |
| OV GPU (UHD 620) | 970.0 ms | 17.4x, but silently fp16 — rejected |

The bf16 row is the finding that mattered: this CPU has no bf16 instructions, so
torch emulates the format. See `vla_openvino.py` for the runtime and
`scripts/export_smolvla_vision.py` for the frozen-IR export and parity gate.

**iGPU rejected on accuracy, not just contention.** The GPU plugin's output
differed from the fp32 reference by `8.94e-01` because it ran fp16 without being
asked to. Forcing `INFERENCE_PRECISION_HINT: f32` would remove most of its
advantage, and the contention risk below still applies on top of that.

**Still worth doing:** OV's `CACHE_DIR` against the ~55.7s cold start, which is
now the most demo-visible cost left, since per-query latency is close to budget.

**Unverified risk, still flagged:** MuJoCo's offscreen GL context and OV's `GPU`
plugin would both be on the same UHD 620. Measure render latency under iGPU load
before ever choosing `GPU` as the OV device. (A contention probe was written for
this but errored out on an unrelated `build_scene` signature mismatch; not
re-run. It matters less now that CPU is the shipped target for accuracy
reasons.)

### Render cost pinned down: a fixed ~220ms/call, not real GPU work

Following the OV question upstream, the render bottleneck is now located much
more precisely than "scene traversal" (the earlier guess above, which is
**wrong** — corrected here):

- `renderer.update_scene(...)` costs **0.0 ms**. All ~220ms is the
  `renderer.render()` GL call itself.
- Only **57 geoms** are actually drawn (`scene.ngeom`), out of 570 in the model.
- Cost is flat at ~215-225ms across **every** knob: 80x60 / 160x120 / 320x240
  render size (tested earlier), and offscreen buffer 320x240 / 640x480 /
  default 1280x720 (tested now).

**Resolved.** It is *not* a driver stall or a software fallback — the context is
real hardware (`GL_VENDOR: Intel`, `GL_RENDERER: Intel(R) UHD Graphics 620`,
`GL_VERSION: 4.6`). Splitting the call apart:

| Stage | Cost |
|---|---|
| `mjr_render`, 0 geoms | 0.3 ms |
| `mjr_render`, 57 geoms | 182 ms |
| `mjr_readPixels` | 1.1 ms |
| `glFinish` | 0.0 ms |

So it *is* geometry cost, just not fill cost. Per geom group: group 0 (13
geoms) 2.5ms, group 2 (44 geoms) **215ms**, group 3 (305 collision primitives)
37ms. Group 2 is the SO-101 **visual meshes** — 360,704 mesh faces total, led by
`wrist_roll_pitch_so101_v2` (53,994 faces), `under_arm_so101_v1` (39,516) and
`base_motor_holder_so101_v1` (37,540), drawn twice for two arms. Vertex/draw
bound, which is exactly why resolution and offscreen-buffer size changed nothing.

Superseding the "next hour should go here" call above: it shouldn't. Per the
correction at the top of this section, rendering is <1% of a query that actually
infers, and `_needs_fresh_frames()` has since removed 49 of every 50 renders
anyway. Decimating those visual meshes remains a genuine ~200ms win available to
the *camera preview* path (`rendering.py`), which renders every frame and does
feel it — but it is not a VLA bottleneck.

## Recommended next step (not implemented here)

The architecturally sound fix is to get rendering + inference off the single
physics/HTTP thread, the same way `simulation_lab/rendering.py` already keeps
JPEG encoding off it: run the VLA's camera render + `select_action()` call in
a background thread (or process) against a *snapshot* of `data.qpos`/`qvel`
(not the live `MjData`, which the physics thread mutates concurrently — see
`autonomy.py`'s `ArmIK` for the existing "separate scratch `MjData`" pattern),
holding `last_action` until the result lands. This wouldn't reduce the
~600ms/query cost, but would stop it from blocking physics stepping and HTTP
responses, which is what actually reads as "lag" in the browser. Flagged
here rather than implemented, since it touches the core engine's threading
model and deserves its own review/testing pass rather than being bundled
into a benchmarking session.
