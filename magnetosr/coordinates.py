"""Pixel-centre coordinates in the original LTE (row, column) convention."""
import torch


def make_coord(shape, ranges=None, flatten=True):
    axes = []
    for i, n in enumerate(shape):
        lo, hi = (-1, 1) if ranges is None else ranges[i]
        r = (hi - lo) / (2 * n)
        axes.append(lo + r + 2 * r * torch.arange(n).float())
    coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
    return coords.reshape(-1, len(shape)) if flatten else coords
