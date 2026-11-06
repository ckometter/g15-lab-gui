"""Descriptive measurement names; the save layer adds the run's date prefix."""
import math
import re


def number_token(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Filename values must be finite.")
    return format(0.0 if value == 0 else value, '.9g').replace('-', 'm').replace('.', 'p')


def axis_token(name):
    name = str(name).replace('μ', 'mu').replace('µ', 'mu').replace('ν', 'nu')
    return re.sub(r'[^\w-]', '', name, flags=re.UNICODE) or 'Vg'


def dgs_filename(*, exposure, acquisitions, axes, field=0, wavelength=1090,
                 positions=None):
    """axes: (name, coordinate array) pairs; positions: X and Y voltage arrays."""
    parts = ['PSDGS' if positions is not None else 'DGS', 'RC',
             number_token(field) + 'T', number_token(exposure) + 's',
             str(int(acquisitions)) + 'a', number_token(wavelength) + 'nm']
    for name, coords in axes:
        if len(coords) > 1:
            parts.extend([axis_token(name), f'{number_token(coords[0])}-{number_token(coords[-1])}V'])
    parts.append('x'.join(str(len(coords)) for _, coords in axes))
    if positions is not None:
        for name, coords in zip(('X', 'Y'), positions):
            span = number_token(coords[0])
            if len(coords) > 1:
                span += '-' + number_token(coords[-1])
            parts.append(name + span + 'V')
        parts.append('x'.join(str(len(coords)) for coords in positions))
    return '_'.join(parts) + '_.h5'
