"""Portable CPU/CUDA inference, preserving historical query normalization."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import models
from .coordinates import make_coord


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = models.make(checkpoint['model'], load_sd=True).to(device).eval()
    return model


def read_sample(path):
    with np.load(path, allow_pickle=False) as data:
        hmi = np.asarray(data['HMIfield'], dtype=np.float32)
        radius = np.asarray(data['Txy'], dtype=np.float32)
        sp = np.asarray(data['SPfield'], dtype=np.float32) if 'SPfield' in data else None
    if hmi.ndim != 2 or hmi.shape != radius.shape:
        raise ValueError(f'{path}: expected equal 2-D HMIfield and Txy arrays')
    if not np.isfinite(hmi).all() or not np.isfinite(radius).all():
        raise ValueError(f'{path}: non-finite input; mask invalid/off-limb data before inference')
    return hmi, radius, sp


@torch.inference_mode()
def predict(model, hmi, radius, shape=(312, 336), chunk_size=8192):
    """Return a Gauss map. Chunks share the full-grid distance minimum."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    device = next(model.parameters()).device
    inp = torch.from_numpy(np.stack([hmi, radius])).float()[None].to(device) / 200.
    coord = make_coord(shape).to(device)[None]
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / shape[0]
    cell[..., 1] *= 2 / shape[1]
    # The original formula used distn.min() over the entire query tensor.
    # Calculating it per chunk would change the output near chunk boundaries.
    if hasattr(model, 'query_distance_min'):
        dist = F.grid_sample(inp[:, 1:2], coord.flip(-1).unsqueeze(1),
                             mode='bilinear', padding_mode='border', align_corners=False)
        model.query_distance_min = dist.min()
    if hasattr(model, 'target_h') and tuple(shape) != (model.target_h, model.target_w):
        raise ValueError('Fixed-grid ablations support only their trained output dimensions')
    model.gen_feat(inp)
    output = [model.query_rgb(coord[:, i:i+chunk_size], cell[:, i:i+chunk_size]).cpu()
              for i in range(0, coord.shape[1], chunk_size)]
    result = torch.cat(output, dim=1).reshape(*shape).numpy() * 200.
    if not np.isfinite(result).all():
        raise ValueError('Model produced non-finite predictions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/pm_lte.pth')
    p.add_argument('--input', type=Path, required=True, help='NPZ file or directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--height', type=int, default=312)
    p.add_argument('--width', type=int, default=336)
    p.add_argument('--chunk-size', type=int, default=8192)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model = load_model(a.checkpoint, a.device)
    files = [a.input] if a.input.is_file() else sorted(a.input.glob('*.npz'))
    if not files:
        p.error('No NPZ samples found')
    a.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        hmi, radius, sp = read_sample(path)
        pred = predict(model, hmi, radius, (a.height, a.width), a.chunk_size)
        arrays = dict(HMIfield=hmi, Txy=radius, pred=pred)
        if sp is not None:
            arrays['SPfield'] = sp
        np.savez_compressed(a.output / path.name, **arrays)
        print(a.output / path.name, flush=True)


if __name__ == '__main__':
    main()
