"""Plot HMI, a generated full-disk reconstruction and a selected zoom region."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hmi', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True, help='Output directory of full_disk.generate')
    p.add_argument('--case', type=Path, help='Case JSON with zoom box and color limits')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sp', type=Path, help='Optional SP L2.1 FITS; creates an additional comparison-map image')
    a = p.parse_args()
    from astropy.io import fits
    with fits.open(a.hmi) as hdus:
        hmi = np.fliplr(hdus[1].data).copy()
    with np.load(a.result/'reconstruction.npz', allow_pickle=False) as z:
        sr = z['SRFfield']
    meta = json.loads((a.result/'manifest.json').read_text())
    case = json.loads(a.case.read_text()) if a.case else {}
    h, w = hmi.shape
    box = case.get('zoom_box', [w//2-100, w//2+100, h//2-100, h//2+100])
    x0, x1, y0, y1 = box
    cy, cx = meta['crop_y'], meta['crop_x']
    sy, sx = meta['scale_y'], meta['scale_x']
    rx0, rx1 = int((x0-cx)*sx), int((x1-cx)*sx)
    ry0, ry1 = int((y0-cy)*sy), int((y1-cy)*sy)
    if not (0 <= rx0 < rx1 <= sr.shape[1] and 0 <= ry0 < ry1 <= sr.shape[0]):
        p.error('Zoom region lies outside the reconstructed crop')
    limit, zoom_limit = case.get('full_disk_limit',200), case.get('zoom_limit',2000)
    fig, axes = plt.subplots(2, 2, figsize=(12,10), constrained_layout=True)
    maps = [hmi, sr, hmi[y0:y1,x0:x1], sr[ry0:ry1,rx0:rx1]]
    titles = ['HMI input', 'Reconstruction (including interpolation fallback)', 'HMI zoom', 'Reconstruction zoom']
    for i,(ax,field,title) in enumerate(zip(axes.flat,maps,titles)):
        lim = limit if i < 2 else zoom_limit
        im = ax.imshow(field, origin='upper', cmap='gray' if i<2 else 'bwr', vmin=-lim,vmax=lim)
        ax.set_title(title); ax.set_axis_off()
        fig.colorbar(im,ax=ax,shrink=.75,label='G')
    axes[0,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='lime'))
    axes[0,1].add_patch(Rectangle((rx0,ry0),rx1-rx0,ry1-ry0,fill=False,color='lime'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output,dpi=160); plt.close(fig)
    if a.sp:
        with fits.open(a.sp) as hdus:
            sp=np.flipud(hdus[4].data).copy()
        fig,ax=plt.subplots(figsize=(7,7),constrained_layout=True)
        im=ax.imshow(sp,origin='upper',cmap='bwr',vmin=-zoom_limit,vmax=zoom_limit)
        ax.set_title('SP comparison observation'); fig.colorbar(im,ax=ax,label='G')
        fig.savefig(a.output.with_name(a.output.stem+'-sp.png'),dpi=160); plt.close(fig)
    print(a.output)


if __name__ == '__main__':
    main()
