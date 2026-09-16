# Live VLA optimization log

What we measured, what we changed, and what we deliberately did **not** change,
while getting the fine-tuned SmolVLA checkpoint running live against the
simulator on the demo laptop.

Measured baseline hardware: **Intel Core i5-8265U + Intel UHD Graphics 620**,
CPU-only PyTorch, no discrete GPU. The deployment target is now an **Intel Core
i5-12400**. All numeric results below remain measurements from the 8265U—not
estimates for the 12400—so rebuild the IR and rerun the commands below on the
deployment machine before making target-hardware latency claims:

```powershell
python scripts/export_smolvla_vision.py      # build the OpenVINO IR, once per checkpoint
python scripts/benchmark_live_vla.py --sim-seconds 6
```

Checkpoint under test: `pjmorales04/duet-smolvla-v2` (7000 steps, batch 32,
trained on seed 1000 — the training partition, per `duet_protocol.py`).
`simulation_lab/vla_task.py` auto-selects `models/duet-smolvla-v2` when present
and falls back to `v1`; `DUET_VLA_CHECKPOINT` overrides for A/B runs.

## Headline result

Same checkpoint, same 6 seconds of simulation, same laptop — measured before and
after the work in this document:

| | Before | After | |
|---|---|---|---|
| Wall time for 6s of sim | 126.47 s | **12.24 s** | **10.3× faster** |
| Real-time factor | 0.05 | **0.49** | |
| Real forward pass | 61 991 ms | **5 252 ms** | **11.8× faster** |
| Amortised cost per query | 1 260.8 ms | **120.3 ms** | vs a 100 ms budget |
| Camera render, per frame | 318.3 ms | 237.5 ms | 97% of renders also eliminated (§3) |

```
render (per frame, n=6):  median=237.5ms  mean=323.4ms
query, cached action (n=58): median=5.8ms
query, real forward pass (n=2, 1 per 50 queries): median=5252ms  max=5527ms
sim_time=6.00s wall_time=12.24s real_time_factor=0.49
amortised per-query cost: 120.3ms vs 100ms budget -> LAGGING
```

**Nothing was traded away for this.** The dominant change runs the same stored
weights through *more* precise arithmetic, not less, and the OpenVINO swap is
exact at action level. §6 gives the parity method, including why a naive
before/after diff on this model is meaningless.

---

## 1. Correctness first: the checkpoint would not load at all

The installed `lerobot==0.3.2` is a different revision from the one the
checkpoint was exported with. Four fixes in `vla_task.py::_load_policy`, none of
which change the model's math — they only get the same arithmetic running on a
newer stack:

| # | Problem | Fix |
|---|---|---|
| 1 | `config.json` carries fields `SmolVLAConfig` no longer accepts (`use_peft`, `pretrained_path`, `rtc_config`, `compile_model`, …) | Filter to the dataclass's real fields before construction |
| 2 | `normalization_mapping` values arrive as plain strings, not `NormalizationMode` members | Construct real enum members (`is`/`isinstance` checks require them) |
| 3 | Installed lerobot bakes `Normalize`/`Unnormalize` buffers into the model and asserts on them; this checkpoint normalizes *outside* the model and never saved those buffers, so `strict=True` loading failed | Force every modality to `IDENTITY`, leaving `update()`'s existing hand-rolled normalization as the single source of truth |
| 4 | `transformers` now needs `accelerate` for the `device_map="auto"` path | Added `accelerate` to `requirements.txt` |

**Verified checkpoint-agnostic:** `duet-smolvla-v2`'s `config.json` is
byte-identical to v1's and it loads through the exact same code path, unchanged.

## 2. A real bug found while profiling: stale action chunks across instructions

SmolVLA is an **action-chunk** policy (`chunk_size=50`): `select_action()` runs a
real forward pass only when its internal 50-action queue is empty, and otherwise
pops an already-predicted action. The policy object is cached across tasks in
`_CACHE`, and nothing reset that queue.

**Effect:** starting a new instruction replayed up to 50 actions predicted for
the *previous* instruction — roughly 5 seconds of the wrong behaviour — before
the new language tokens ever reached the model. Fixed with a `policy.reset()` in
`SmolVLATask.start()`. This is a behavioural correctness fix, not a speed fix,
and it matters most in exactly the case a judge will hit: typing a second
instruction after the first one.

## 3. Shipped optimization: render only the frames the policy actually reads

The same chunk-queue behaviour exposed the single biggest waste. `vla_task.py`
rendered **two camera frames on every query**, but 49 out of every 50 queries
just pop a cached action and never look at the pixels.

At ~220ms per rendered frame, a 6-second run was spending **~26.4s rendering
frames that nothing read**.

`_needs_fresh_frames()` now renders only on the query that actually consumes the
images, reusing the cached tensors otherwise.

| | Renders per 6s sim run | Render wall-time |
|---|---|---|
| Before | 120 | ~26.4 s |
| After | 4 | ~0.9 s |

**~97% of rendering work eliminated, bit-identical policy output** — the skipped
frames were provably discarded (they pass through `normalize_inputs`, which this
checkpoint forces to IDENTITY, and are then dropped). The check is guarded in
`try/except`: if lerobot's internals ever move, it falls back to rendering every
query — slower, never wrong.

## 4. Root-cause analysis: where the time actually goes

Two earlier conclusions in `simulation_lab/PERF_NOTES.md` were wrong and are
corrected here. Profiling in stages mattered:

**The "~20ms inference" number was measuring the queue-pop path, not a forward
pass.** Forcing real chunk refills:

| Operation | Cost (as originally measured, bf16) |
|---|---|
| Query that pops a cached action | **12 ms** |
| Query that runs a real forward pass | **62.5 s** (54.9 / 60.7 / 62.5 / 63.2 s across runs) |
| Camera render, per frame | 220 ms |
| Cold checkpoint load | 53–72 s |

**Rendering was never the real bottleneck — a forward pass was ~140× more
expensive than the two frames feeding it.** Fixing the render waste (§3) was
still correct and worth it, but it is not what stood between this laptop and a
real-time demo.

### Inside the forward pass

`num_steps` (flow-matching denoising) sweep: `num_steps=1` → 56.2s,
`num_steps=5` → 62.8s. So the denoising loop was ~1.6s/step, about **10%** of
the total. The remaining **~88% was VLM vision encoding** — two 512×512 images
through a 16-layer SmolVLM2-500M on a 4-core CPU.

Localising the cost to the vision tower is what made §5 and §6 findable: it
pointed at one contiguous subgraph, which turned out to be both the thing
running in an unsupported dtype and the thing worth handing to OpenVINO. After
both changes the same pass is **5 252 ms**, of which roughly 61% is still the
(now OpenVINO, now fp32) vision tower — so it remains the right target for §8.

### Inside the 220ms render

Measured with the offscreen context split apart:

| Stage | Cost |
|---|---|
| `update_scene` (CPU scene build) | **0.0 ms** |
| `mjr_render`, 0 geoms | 0.3 ms |
| `mjr_render`, 57 geoms | 182 ms |
| `mjr_readPixels` | 1.1 ms |
| `glFinish` | 0.0 ms |

The GL context is **real hardware** (`GL_RENDERER: Intel(R) UHD Graphics 620`,
GL 4.6) — not a software fallback, and readback is not the problem. Cost is
entirely per-geometry: geom **group 2 alone (the SO-101 visual meshes) is 215ms
of it**, against 360,704 mesh faces, while group 3's 305 collision primitives
cost only 37ms. Cost is flat across render size (80×60 ≡ 320×240) and offscreen
buffer size (320×240 ≡ 1280×720), confirming it is vertex/draw-bound, not
fill-bound.

## 5. The dominant win: the CPU cannot execute the checkpoint's dtype

The checkpoint ships **bfloat16**. The demo CPU is an **i5-8265U (Whiskey Lake,
2018)**, which predates AVX512-BF16 and AMX and has **no bf16 instructions at
all**. PyTorch therefore *emulates* every bf16 operation — unpack to fp32,
compute, repack — on a ~450M-parameter model, 60 times a run.

Measured on the vision tower, **identical weights**, one 512×512 image:

| dtype | Latency | |
|---|---|---|
| bfloat16 (as shipped) | 16 906.7 ms | |
| float32 | 1 904.4 ms | **8.9× faster** |

And on the complete forward pass, via a one-line `policy.float()`:

| | Forward pass | |
|---|---|---|
| bfloat16 | 46 023 ms | |
| float32 | 5 226 ms | **8.8× faster** |

This is the single largest change in this document, it costs one line, and it
makes the model **more** accurate rather than less — bf16 carries ~3 decimal
digits, fp32 carries ~7, and the stored weights are unchanged either way.
`DUET_VLA_BF16=1` restores the original dtype for comparison.

Safe because every dtype cast inside lerobot's SmolVLA is *relative* to the
weight dtype (`.to(dtype=layer.self_attn.q_proj.weight.dtype)`), so the cast
propagates cleanly; only `smolvlm_with_expert.py`'s load-time
`torch_dtype="bfloat16"` pins it, and that has already run by then.

## 6. OpenVINO: shipped, not just measured

`openvino==2026.3.0` was already a dependency (`rgb_servo_openvino.py`,
`primitive_policy.py`, `bottle_refinement_runtime.py`, the mug-keypoint export),
but **the VLA path was the one component that never used it**. It does now.

**What runs in OpenVINO:** the vision tower — `connector(vision_model(pixels))`,
exactly what `SmolVLMWithExpertModel.embed_image()` computes, and ~61% of the
remaining forward pass. **What does not:** the language model, action expert,
flow-matching denoiser, image resize/normalization, tokenizer and action
post-processing all remain explicit PyTorch. That boundary is stated in
`simulation_lab/vla_openvino.py`'s docstring and matches the contract language
used by the other OpenVINO adapters in this repo.

| Vision tower, one image | Latency | vs shipped bf16 |
|---|---|---|
| PyTorch bfloat16 | 16 906.7 ms | 1.0× |
| PyTorch fp32 | 1 904.4 ms | 8.9× |
| **OpenVINO CPU FP32** | **1 605.6 ms** | **10.5×** |
| OpenVINO GPU (UHD 620) | 970.0 ms | 17.4× — **rejected**, see below |

Exported by `scripts/export_smolvla_vision.py`, which follows the same discipline
as `export_mug_keypoints.py`: frozen IR, `compress_to_fp16=False`,
`PERFORMANCE_HINT: LATENCY`, explicit `INFERENCE_PRECISION_HINT: f32`, SHA-256 of
the IR, checkpoint, exporter and runtime source recorded in `parity.json`, and a
hard failure if parity regresses. Verified `max|diff| = 5.34e-05` against the
identical fp32 weights. `models/` is gitignored, so the IR is a reproducible
build artifact, not a committed blob:

```powershell
python scripts/export_smolvla_vision.py
```

The runtime binds it by shadowing `embed_image` on that **one policy instance**,
and falls back to stock PyTorch if the IR is missing, the device is gone, or
`parity.json` does not record a pass — slower, never silently wrong.
`DUET_VLA_OPENVINO=0` forces PyTorch; `DUET_VLA_OV_DEVICE` selects the plugin.
The active runtime is reported in the task snapshot as
`policy_details.neural_runtime`, the same way `primitive_policy.py` does it.

### Why the iGPU was rejected despite being fastest

OpenVINO's GPU plugin returned 970.0 ms — but with `max|diff| = 8.94e-01`
against the fp32 reference, because it silently ran **fp16**. That is not a
speedup, it is a different model. Forcing `INFERENCE_PRECISION_HINT: f32` would
erase most of the advantage, and the UHD 620 is already driving MuJoCo's
offscreen GL context, so inference there would contend with rendering. CPU is
the shipped target, consistent with `package_spoon_release.py`'s existing
`neural_execution_devices == ['CPU']` gate.

### Parity method: separating real error from sampling noise

SmolVLA's `sample_noise()` uses **unseeded** `torch.normal`, so two identical
calls already disagree. A naive before/after diff showed 17.5% and meant
nothing. Seeding the flow-matching noise and measuring a same-dtype control
separates the two:

| Comparison | Action difference |
|---|---|
| **Control** — bf16 vs bf16, different noise seed | 21.60% |
| **Control** — fp32 vs fp32, different noise seed | 20.73% |
| **Effect** — bf16 vs fp32, *same* seed | **0.72%** |
| **Effect** — PyTorch fp32 vs OpenVINO, *same* seed | **0.00%** |

The dtype change moves the action ~30× less than the model's own stochasticity,
and the OpenVINO swap is exact at action level. Any comparison of this model
that does not pin the noise seed is measuring sampling variance.

## 7. Tried and rejected

Recorded so nobody re-runs them:

- **`load_vlm_weights=False`** (skip downloading pretrained SmolVLM weights that
  `load_state_dict` overwrites anyway). **Measured slower: ~147s vs ~55-70s cold
  start.** The `True` path uses `low_cpu_mem_usage=True` and streams bytes into
  place; the `from_config` fallback fully random-initializes ~450M parameters
  first. Reverted.
- **Lower render resolution.** No effect — render is vertex-bound, not
  fill-bound (§4). 80×60 costs the same as 320×240.
- **Disabling reflections / skybox / fog.** A genuine **2.3× render win**
  (274ms → 119ms) that we chose *not* to ship: `collect_dinner_learning.py`
  renders the training frames with exactly the same flags as `vla_task.py`
  (`geomgroup[3:]=0`, shadows off, 320×240, `offsamples=0`). Changing them at
  inference time only would introduce a train/inference visual mismatch and cost
  task success to buy speed we don't need now that rendering is 1% of the
  budget. **This is the right change to bake into the next dataset
  regeneration**, where training and inference move together and it is free.
- **INT8 dynamic quantization** — previously tried and rejected in-tree
  (see the `ponytail:` note in `vla_task.py`).

Already in place before this pass and confirmed to be real wins: shadows
disabled per-frame (~45%), `offsamples=0` at renderer construction,
`num_steps` 10→5, and a bounded torch thread count.

## 8. Where the remaining time goes

The forward pass is now **5 252 ms**, split roughly:

| Stage | Cost | Runtime |
|---|---|---|
| Vision tower, 2 × 512×512 images | ~3 217 ms (61%) | OpenVINO CPU FP32 |
| LM prefix + action expert + 5 denoising steps | ~2 035 ms (39%) | PyTorch CPU FP32 |

Amortised over a 50-action chunk that is 120.3 ms per query against a 100 ms
budget — **within 20% of real time, from 12.6× over it.**

### INT8 quantization: implemented, measured, and rejected

The obvious next lever was **INT8 post-training quantization via NNCF**, which
normally buys 2–3× on CPU. It is implemented in
`scripts/quantize_smolvla_vision.py`, calibrated on 48 real dinner-scene renders
drawn from **training-partition seeds** (enforced through
`duet_protocol.require_partition`, since collecting calibration statistics fits
parameters). Unlike the dynamic-quantization attempt in §7 it quantizes the
*exported IR*, so it does not hit the ONEDNN dtype failure that killed that one.

It works, and it is **slower**:

| Vision tower | Latency | Model size |
|---|---|---|
| OpenVINO CPU FP32 | 1 972.0 ms | 392.9 MB |
| OpenVINO CPU INT8 | **2 787.0 ms** (**0.71×** — 41% *slower*) | 99.2 MB |

**Cause, verified not assumed:** `torch.backends.cpu.get_cpu_capability()`
reports **AVX2**. This CPU has no AVX512 and therefore no **VNNI / DL Boost**,
the instruction set INT8 inference depends on. Without it OpenVINO emulates
int8 dot products, and the dequantize/requantize overhead costs more than a
well-tuned AVX2 fp32 GEMM. Tower output also moved 4.5% relative.

This is the **same finding as §5, in a different format**: on an AVX2-only
2018 CPU, fp32 is not a compromise — it is the only numeric format the machine
is actually fast at. bf16 and int8 both lose, for the same underlying reason.
The script is kept, because on any VNNI-capable Intel CPU (Ice Lake or newer)
this is expected to flip and become the largest remaining win; `parity_int8.json`
records the full result so the decision is re-checkable on other hardware.

### What is left

- OpenVINO's `CACHE_DIR` against the **55.7 s cold start**, now the most
  demo-visible cost, since per-query latency is close to budget.
- The 20% over budget is small enough that the honest fix is architectural:
  moving render + inference off the physics thread (deliberately unimplemented,
  see `PERF_NOTES.md`). That hides the remaining gap rather than removing it,
  which is the right trade for a demo but should be labelled as such.
- Regenerating the dataset with reflections/skybox/fog disabled (§7) would make
  the render change free, though rendering is no longer a meaningful cost.

## 9. Honest conclusion

> **(superseded)** An earlier revision of this document concluded: *"This laptop
> cannot run the live VLA at real time, and no amount of render tuning changes
> that… a real-time factor ceiling around 0.08. Even a perfect 2.8× OpenVINO
> port lands near 0.2."* That was correct about render tuning and wrong about
> the ceiling, because it assumed the bf16 cost was intrinsic rather than an
> emulation artifact of this specific CPU (§5). The measured factor is **0.49**.

What the work above delivers:

- The checkpoint loads and runs correctly at all (§1), on v1 and v2 alike.
- A real behavioural bug is fixed (§2).
- Rendering is no longer a meaningful cost (§3, ~97% eliminated).
- The remaining cost was precisely located (§4), which is what made §5 findable.
- **A 10.3× end-to-end speedup with no accuracy cost** (§5, §6), of which the
  decisive part was recognising that the demo CPU cannot execute the dtype the
  checkpoint ships in.
- The VLA path now runs on OpenVINO, with the same export discipline, parity
  gating and CPU-FP32 execution contract as every other OpenVINO component in
  this repo (§6).

The live VLA is now genuinely demoable on this laptop: one ~5 s forward pass,
then 5 seconds of smooth 50-action playback, repeating. The expert-teacher
gallery remains the path for side-by-side comparison.

---

*Engineering detail, including the two superseded conclusions and the reasoning
behind each fix, is in [`simulation_lab/PERF_NOTES.md`](simulation_lab/PERF_NOTES.md).*
