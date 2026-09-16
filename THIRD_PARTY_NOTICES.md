# Third-party notices

**Original Talos code, model weights and procedural dinner assets use [MIT](LICENSE). Third-party components retain their own licenses.** The root MIT license and repository badge describe Talos's original contributions; they do not relicense the components below.

## Bundled assets and archived source

- SO-101 robot assets use [Apache-2.0](simulation_lab/assets/so101/LICENSE). Their original license, source revision, hashes and modification notices are retained in [simulation_lab/NOTICE.md](simulation_lab/NOTICE.md) and [asset provenance](simulation_lab/assets/so101/provenance.json). This includes the simplified visual meshes derived from the originals. These notices and license files accompany the hosted Space too.
- Archived spatial-policy research includes a modified Apache-2.0 CLIPort component and MIT OpenAI CLIP source. Their licenses, attribution and modifications remain in the [archived component](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/third_party/talos_cliport). No pretrained Talos spatial policy was completed, and these components are not the selected demo controller. The upstream GPL U-Net helper was excluded.

## Direct Python dependencies

This inventory covers the directly declared packages in the pinned [local CPU requirements](https://github.com/jannissio/talos-ai-infra/blob/main/requirements.txt), [hosting requirements](https://github.com/jannissio/talos-ai-infra/blob/main/hosting/requirements.txt) and [optional training requirements](https://github.com/jannissio/talos-ai-infra/blob/main/training/requirements.txt). Package links identify the published distributions; full license and copyright notices remain in those distributions.

| Package | Upstream license | Declared use |
| --- | --- | --- |
| [FastAPI 0.141.1](https://pypi.org/project/fastapi/0.141.1/) | MIT | Local app, training environment |
| [Uvicorn 0.52.4](https://pypi.org/project/uvicorn/0.52.4/) | BSD-3-Clause | Local app, training environment |
| [MuJoCo 3.12.0](https://pypi.org/project/mujoco/3.12.0/) | Apache-2.0 | Physics in all environments |
| [NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/) | BSD-3-Clause | All environments |
| [Pillow 12.3.0](https://pypi.org/project/pillow/12.3.0/) | MIT-CMU | Local app, hosted demo |
| [PyOpenGL 3.1.10](https://pypi.org/project/PyOpenGL/3.1.10/) | BSD-style core; additional [component notices](https://github.com/mcfletch/pyopengl/blob/master/license.txt) | Rendering in all environments |
| [Safetensors 0.7.0](https://pypi.org/project/safetensors/0.7.0/) | Apache-2.0 | Local app, hosted demo |
| [Packaging 26.0](https://pypi.org/project/packaging/26.0/) | Apache-2.0 OR BSD-2-Clause | Local CPU environment |
| [OpenVINO 2026.3.0](https://pypi.org/project/openvino/2026.3.0/) | Apache-2.0 | Local and hosted CPU inference |
| [OpenVINO Telemetry 2025.2.0](https://pypi.org/project/openvino-telemetry/2025.2.0/) | Apache-2.0 | Local CPU environment |
| [PyTorch 2.8.0](https://pypi.org/project/torch/2.8.0/) | BSD-3-Clause | All environments; CPU/CUDA builds have additional bundled notices |
| [TorchVision 0.23.0](https://pypi.org/project/torchvision/0.23.0/) | BSD-3-Clause | Optional training |
| [TorchCodec 0.7.0](https://pypi.org/project/torchcodec/0.7.0/) | BSD-3-Clause | Optional training |
| [LeRobot 0.6.1](https://pypi.org/project/lerobot/0.6.1/) | Apache-2.0 | Optional training |
| [Gradio 6.27.0](https://pypi.org/project/gradio/6.27.0/) | Apache-2.0 | Hosted interface |

This table summarizes direct package licenses. Transitive dependencies, bundled native libraries, graphics drivers and system packages retain their respective license and notice files. For example, NumPy and PyTorch distributions include additional library notices, and any separately installed FFmpeg used with TorchCodec has its own build-dependent terms. Those distributions are installed from upstream; Talos's MIT license does not replace their terms.

## Submission artwork

The current cover is an AI-generated illustration based on Talos's own simulation image. It is artwork rather than measurement evidence. Its generation brief is recorded in the [submission guide](https://github.com/jannissio/talos-ai-infra/blob/main/submission/README.md).
