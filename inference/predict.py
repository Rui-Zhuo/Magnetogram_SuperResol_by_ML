"""CPU/CUDA inference with explicit patch-center and legacy PM behavior."""
import argparse
import copy
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import models
from models.coordinates import make_coord


def load_model(path, device='cpu', position_mode=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    spec = copy.deepcopy(checkpoint['model'])
    if position_mode is not None:
        if spec['name'] != 'lte-warp':
            raise ValueError('position_mode overrides are supported only for the LTEW decoder')
        recorded_mode = spec['args'].get('position_mode', 'legacy_query_min')
        if position_mode != recorded_mode:
            warnings.warn(f'Overriding checkpoint PM from {recorded_mode} to {position_mode} changes '
                          'predictions. This override is not a verified reproduction of the saved '
                          'paper results; the retained checkpoint tensors are unchanged.',
                          UserWarning, stacklevel=2)
        spec['args']['position_mode'] = position_mode
    model = models.make(spec, load_sd=True).to(device).eval()
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
    """Return a Gauss map; both PM versions are independent of query chunking."""
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
    if getattr(model, 'position_mode', None) == 'legacy_query_min' and model.use_pm:
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
    p.add_argument('--config', type=Path, help='Test YAML; explicit options override its values')
    p.add_argument('--checkpoint')
    p.add_argument('--position-mode', choices=['patch_center', 'legacy_query_min'],
                   help='Explicit LTEW PM override; reusing a legacy checkpoint changes predictions')
    p.add_argument('--input', type=Path, help='NPZ file or directory')
    p.add_argument('--output', type=Path)
    p.add_argument('--device')
    p.add_argument('--height', type=int)
    p.add_argument('--width', type=int)
    p.add_argument('--chunk-size', type=int)
    p.add_argument('--threads', type=int)
    a = p.parse_args()
    defaults = dict(checkpoint='pre-trained/pm_ltew_cc.pth', device='cpu',
                    height=312, width=336, chunk_size=8192, threads=4, position_mode=None)
    if a.config:
        import yaml
        config = yaml.safe_load(a.config.read_text(encoding='utf-8'))
        allowed = set(defaults) | {'input', 'output', 'model_name'}
        if not isinstance(config, dict) or set(config)-allowed:
            p.error('Test configuration must contain only inference options and model_name')
        defaults.update(config)
    for key, value in defaults.items():
        if key != 'model_name' and getattr(a, key, None) is None:
            setattr(a, key, value)
    if a.input is None or a.output is None:
        p.error('Provide input/output in --config or as command options')
    a.input, a.output = Path(a.input), Path(a.output)
    torch.set_num_threads(a.threads)
    model = load_model(a.checkpoint, a.device, a.position_mode)
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
