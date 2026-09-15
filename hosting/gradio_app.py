"""Compact hosted Talos demo; local server and credentials are not imported."""
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', str(Path(tempfile.gettempdir()) / 'talos-gradio'))

import gradio as gr
if os.environ.get('SPACE_ID'):
    # Register the actual GPU-dependent function required by the free tier.
    from hosting.gpu_inference import generate_trajectory
else:
    generate_trajectory = None
import torch
from simulation_lab.public_trial import load_checkpoints, run_trial
from simulation_lab.scene import CAMERAS
from simulation_lab.learned_skills import SKILL_LABELS
from simulation_lab.hosted_worker import TrialWorker

torch.set_num_threads(2)
SUITE = os.environ.get('TALOS_DEMO_SUITE')
AVAILABLE = load_checkpoints(SUITE)
VISUAL_MUG_AVAILABLE=(ROOT/'models/dinner_visual_mug_v1/profile.json').is_file()
WIDE_BOTTLE_AVAILABLE=(ROOT/'models/bottle_wide_v1/profile.json').is_file()
JOBS = {}
JOBS_LOCK = Lock()


def simulate(instruction, seed, controller, bottle_start, camera, inference, request: gr.Request):
    if inference not in ('cpu', 'gpu'):
        raise gr.Error('Choose one of the offered inference devices.')
    if controller in ('learned_visual','learned_wide') and inference!='cpu':
        raise gr.Error('Live visual control uses CPU · OpenVINO. Select CPU to run this controller.')
    use_gpu = controller == 'learned' and inference == 'gpu'
    if use_gpu and generate_trajectory is None:
        raise gr.Error('Shared GPU inference is unavailable in this local interface.')
    job = TrialWorker((instruction, seed, controller, bottle_start, camera),
                      suite=SUITE, cache_dir=os.environ['GRADIO_TEMP_DIR'],
                      max_wall_seconds=420,
                      gpu_callback=generate_trajectory if use_gpu else None)
    with JOBS_LOCK:
        JOBS[request.session_hash] = job
    try:
        for image, status, result in job.frames():
            result = dict(result)
            result['inference_runtime'] = ('PyTorch CUDA via shared ZeroGPU' if use_gpu else 'OpenVINO CPU') if controller in ('learned','learned_visual','learned_wide') else 'Programmed controller'
            yield image, status, result
    finally:
        with JOBS_LOCK:
            if JOBS.get(request.session_hash) is job:
                JOBS.pop(request.session_hash)
        job.close()


def cancel_trial(request: gr.Request):
    with JOBS_LOCK:
        job = JOBS.pop(request.session_hash, None)
    if job is not None:
        job.stop()
    return 'Cancelled. Run fresh trial to start again.', {'status':'cancelled'}


with gr.Blocks(title='Talos · Dinner robotics', delete_cache=(3600, 3600)) as demo:
    gr.Markdown('# Talos\n### Give an instruction. Watch a fresh physical simulation.')
    gr.Markdown('Two SO-101 arms share a dinner table. This compact demo starts a new scene for each trial. '
                'Learned control uses camera images, neural motor targets and motor feedback; a separate physical monitor checks outcomes. '
                + ('Live mug vision adds stereo correction during late mug placement. Other skills use initial camera observations. ' if VISUAL_MUG_AVAILABLE else '') +
                'Programmed control uses exact simulator state. Supported language and starting regions are limited.')
    with gr.Row():
        with gr.Column(scale=1):
            instruction = gr.Textbox(label='Instruction', value='Set the table' if VISUAL_MUG_AVAILABLE else 'Place the bottle', max_lines=4)
            seed = gr.Number(label='Scene seed', value=42, precision=0, minimum=0, maximum=2147483647)
            controllers=[('Learned dinner · initial vision','learned'),('Programmed physical skills','programmed')]
            if VISUAL_MUG_AVAILABLE:controllers.insert(0,('Learned dinner · live mug vision','learned_visual'))
            if WIDE_BOTTLE_AVAILABLE:controllers.insert(0,('Learned dinner · wider bottle + live mug','learned_wide'))
            controller = gr.Radio(controllers, value='learned_visual' if VISUAL_MUG_AVAILABLE else 'learned', label='Controller')
            devices = [('CPU · OpenVINO', 'cpu')]+([('Shared GPU · queue may be unavailable', 'gpu')] if generate_trajectory is not None else [])
            inference = gr.Radio(devices, value='cpu', label='Learned inference',
                                 info='The small learned models run on CPU without waiting for a shared GPU.')
            starts=[('Upright, standard region', 'upright'), ('Upright, farther left', 'wide_left')]
            if WIDE_BOTTLE_AVAILABLE:starts.append(('Wider region · varies with seed','wide_rectangle'))
            bottle_start = gr.Dropdown(starts, value='upright', label='Bottle starting region')
            camera = gr.Dropdown([(label, name) for name, label in CAMERAS.items()], value='opposite', label='Camera')
            with gr.Row():
                run = gr.Button('Run fresh trial', variant='primary')
                cancel = gr.Button('Cancel trial')
        with gr.Column(scale=2):
            view = gr.Image(label='Live simulation', type='numpy', format='jpeg', interactive=False, buttons=['fullscreen'])
            status = gr.Textbox(label='Progress and outcome', interactive=False)
    with gr.Accordion('Trial evidence', open=False):
        report = gr.JSON(label='Physical outcome')
    gr.Markdown('**Loaded learned skills:** ' + ', '.join(SKILL_LABELS.get(s, s) for s in AVAILABLE) + '.\n\n'
                '**Try:** “place the bottle”; “set the table”; “pass the bottle to the right arm”. '
                'Use “set the table” for live mug vision; choose initial vision for individual mug commands. '
                'Additional learned instructions require their corresponding loaded models. '
                'Unsupported commands and scenes are refused; a failed grasp is reported as a failure. '
                'Table-supported bottle relays release before the second arm grasps. '
                'This page has no microphone capture; voice and six simultaneous camera views are available in the local browser lab.')
    event = run.click(simulate, [instruction, seed, controller, bottle_start, camera, inference], [view, status, report],
                      concurrency_limit=1, concurrency_id='simulation', api_name='simulate', api_visibility='undocumented')
    cancel.click(cancel_trial,
                 outputs=[status, report], cancels=[event], queue=False, api_name='cancel', api_visibility='undocumented')

if __name__ == '__main__':
    demo.queue(max_size=8).launch(server_name='0.0.0.0' if os.environ.get('SPACE_ID') else '127.0.0.1',
                                server_port=int(os.environ.get('PORT', '7860')), share=False,
                                show_error=False, max_file_size='1mb', run_history=False)
