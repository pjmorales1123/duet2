"""Conservative preflight checks; never delete user artifacts automatically."""
import shutil
from pathlib import Path

GIB = 1024 ** 3


def require_space(destination, additional_bytes, reserve_bytes=10 * GIB):
    path = Path(destination).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < additional_bytes + reserve_bytes:
        raise OSError(f'Insufficient disk headroom: {free/GIB:.1f} GiB free; '
                      f'{additional_bytes/GIB:.1f} GiB estimated writes plus {reserve_bytes/GIB:.1f} GiB reserve required.')
    return {'free_bytes': free, 'estimated_additional_bytes': additional_bytes,
            'reserve_bytes': reserve_bytes}
