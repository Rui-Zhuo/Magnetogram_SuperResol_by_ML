"""Show the input, reference, prediction and residual with honest units."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import bicubic, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('examples/data'))
    p.add_argument('--predictions', type=Path, default=Path('examples/predictions'))
    p.add_argument('--output', type=Path, default=Path('examples/figures'))
    p.add_argument('--model-name', default='PM-LTEW-CC',
                   choices=['PM-LTEW-CC', 'PM-LTEW', 'LTEW-CC', 'LTEW', 'PM-RCAN-CC', 'PM-SRCNN-CC'])
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(a.predictions.glob('*.npz')):
        with np.load(a.data/path.name) as d: hmi, gt = d['HMIfield'], d['SPfield']
        with np.load(path) as d: pred = d['pred'].reshape(gt.shape)
        fields = [hmi, bicubic(hmi, gt.shape), gt, pred, pred-gt]
        fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
        for ax, field, title in zip(axes, fields, ['HMI', 'Bicubic', 'Hinode/SP', a.model_name, 'Prediction - SP']):
            limit = 500 if title == 'Prediction - SP' else 1500
            im = ax.imshow(field, origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit,
                           extent=[0, 100, 0, 100], aspect='equal')
            ax.set_title(title); ax.set_xlabel('Patch x (arcsec)')
            fig.colorbar(im, ax=ax, orientation='horizontal', label='G', shrink=.85)
        axes[0].set_ylabel('Patch y (arcsec)')
        metrics = evaluate(pred, gt)
        fig.suptitle(f'{path.stem} | held-out test sample | RMSE {metrics["RMSE_G"]:.1f} G | CC {metrics["CC"]:.3f}')
        fig.savefig(a.output/(path.stem+'.png'), dpi=160); plt.close(fig)


if __name__ == '__main__': main()
