"""Classify measured inference reports without inferring device identity from IDs."""


def all_inference_devices_are_intel(benchmarks):
    if not benchmarks:
        return False
    for report in benchmarks.values():
        name = str(report.get('name','')).casefold()
        if report.get('parity_passed') is not True or 'intel' not in name:
            return False
        if any(vendor in name for vendor in ('nvidia','amd','llvmpipe')):
            return False
    return True
