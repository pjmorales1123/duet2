"""Record actual Intel inference/rendering hardware without personal identifiers.

Run on the final target. An AMD/NVIDIA result cannot satisfy the Intel check.
Legacy Intel hardware is reported separately from Core Ultra Series 2/3; this
script cannot decide organizer eligibility or resolve conflicting instructions.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space


def cpu_name():
    if sys.platform == 'win32':
        return subprocess.check_output(['powershell', '-NoProfile', '-Command',
            '(Get-CimInstance Win32_Processor | Select-Object -First 1).Name'], text=True).strip()
    path = Path('/proc/cpuinfo')
    if path.exists():
        match = re.search(r'^model name\s*:\s*(.+)$', path.read_text(), re.MULTILINE)
        if match: return match.group(1)
    return platform.processor()


def inspect():
    import mujoco
    import openvino as ov
    from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
    core = ov.Core()
    cpu = cpu_name()
    devices = []
    for device in core.available_devices:
        devices.append({'id': device, 'name': str(core.get_property(device, 'FULL_DEVICE_NAME'))})
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=120, width=160)
    try:
        renderer.update_scene(data); renderer.render()
        graphics = {key: (glGetString(code) or b'').decode('utf-8', errors='replace')
                    for key, code in [('vendor', GL_VENDOR), ('renderer', GL_RENDERER), ('version', GL_VERSION)]}
    finally: renderer.close()
    intel_cpu = 'intel' in cpu.lower()
    intel_graphics = 'intel' in (graphics['vendor']+' '+graphics['renderer']).lower()
    match = re.search(r'core.*?ultra.*?\b([23]\d\d)[a-z]*\b', cpu, re.IGNORECASE)
    series = int(match.group(1)[0]) if match else None
    return {'schema': 'talos.intel-target-inspection.v1', 'cpu': cpu,
            'os': platform.system(), 'os_release': platform.release(), 'python': platform.python_version(),
            'mujoco': mujoco.__version__, 'openvino': ov.__version__, 'openvino_devices': devices,
            'opengl': graphics, 'intel_cpu': intel_cpu, 'intel_opengl_renderer': intel_graphics,
            'core_ultra_series': series, 'strict_written_hardware_check': bool(intel_cpu and series in (2, 3)),
            'all_intel_cpu_and_graphics': bool(intel_cpu and intel_graphics),
            'physical_policy_run': 'Required separately; device discovery does not demonstrate task success.',
            'eligibility': 'Record actual hardware. Organizer confirmation remains necessary for legacy Intel under the livestream clarification.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inspect-only', action='store_true', help='Report the machine without requiring Intel/Core Ultra.')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    require_space(args.output, 8*1024**2)
    report = inspect()
    require_space(args.output, 8*1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if args.inspect_only or (report['strict_written_hardware_check'] and report['all_intel_cpu_and_graphics']) else 1)
