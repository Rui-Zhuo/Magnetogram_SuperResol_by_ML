"""Display a downloaded FITS map, a paired NPZ, or a generated full-disk NPZ."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='FITS or NPZ array')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hdu', type=int, default=1, help='FITS extension: HMI=1, SP L2.1 target=4')
    p.add_argument('--key', help='NPZ array key; default HMIfield')
    p.add_argument('--limit', type=float, default=200, help='Symmetric display limit in G')
    p.add_argument('--title', default='Magnetogram')
    a = p.parse_args()
    if a.limit <= 0: p.error('--limit must be positive')
    if a.input.suffix.lower() == '.npz':
        with np.load(a.input, allow_pickle=False) as data:
            field = np.array(data[a.key or 'HMIfield'])
    else:
        from astropy.io import fits
        with fits.open(a.input) as hdus:
            field = np.array(hdus[a.hdu].data)
    if field.ndim != 2: p.error('Selected data must be a two-dimensional magnetic field')
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    im = ax.imshow(field, origin='lower', cmap='RdBu_r', vmin=-a.limit, vmax=a.limit,
                   interpolation='nearest')
    ax.set(xlabel='Array column (pixel)', ylabel='Array row (pixel)', title=a.title)
    fig.colorbar(im, ax=ax, shrink=.8, label='Magnetic field (G)')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
