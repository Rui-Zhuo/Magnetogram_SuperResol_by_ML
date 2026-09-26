"""Fixed-grid PM-RCAN-CC ablation for magnetogram super-resolution.

Dependencies: Python 3.6.13 and PyTorch 1.10.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from magnetosr import models
from .models import register


@register("pm-rcan-cc")
class PMRCANCC(nn.Module):
    """Replace LTEW queries with a fixed-grid RCAN reconstruction head."""

    def __init__(
        self,
        encoder_spec,
        target_h=312,
        target_w=336,
        use_pm=True,
        distance_scale=960.0,
        position_input_scale=200.0,
        pm_min=0.5,
        pm_max=1.5,
    ):
        super(PMRCANCC, self).__init__()
        self.encoder = models.make(encoder_spec)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.use_pm = bool(use_pm)
        self.distance_scale = float(distance_scale)
        self.position_input_scale = float(position_input_scale)
        self.pm_min = float(pm_min)
        self.pm_max = float(pm_max)

        self.reconstruction = nn.Conv2d(
            self.encoder.out_dim, 1, kernel_size=3, padding=1
        )
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        nn.init.normal_(self.reconstruction.weight, mean=0.0, std=1e-3)
        if self.reconstruction.bias is not None:
            nn.init.zeros_(self.reconstruction.bias)

        self.inp = None
        self.prediction_map = None

    def _fixed_grid_map(self, inp):
        features = self.encoder(inp)
        features = F.interpolate(
            features,
            size=(self.target_h, self.target_w),
            mode="bicubic",
            align_corners=False,
        )
        correction = self.reconstruction(features)

        field = F.interpolate(
            inp[:, 0:1],
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )
        if self.use_pm:
            if inp.shape[1] < 2:
                raise ValueError("PM requires the Txy input channel")
            # Use the same patch-center radius ratio as PM-LTEW-CC.
            distance_norm = (
                inp[:, 1:2, inp.shape[-2] // 2, inp.shape[-1] // 2]
                * self.position_input_scale / self.distance_scale
            ).view(-1, 1, 1, 1)
            modulation = 1.0 - self.gamma * distance_norm + self.beta
            modulation = modulation.clamp(self.pm_min, self.pm_max)
            field = field * modulation
        return correction + field

    def gen_feat(self, inp):
        """Build the fixed prediction map for batched evaluation."""
        self.inp = inp
        self.prediction_map = self._fixed_grid_map(inp)
        return self.prediction_map

    def query_rgb(self, coord, cell=None):
        """Sample the fixed prediction map at requested coordinates."""
        del cell
        if self.prediction_map is None:
            raise RuntimeError("gen_feat must be called before query_rgb")
        return F.grid_sample(
            self.prediction_map,
            coord.flip(-1).unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

    def forward(self, inp, coord=None, cell=None):
        del cell
        prediction_map = self._fixed_grid_map(inp)
        prediction = prediction_map.permute(0, 2, 3, 1).contiguous().view(
            inp.shape[0], -1, 1
        )
        if coord is not None and coord.shape[1] != prediction.shape[1]:
            raise ValueError(
                "Fixed-grid output has {} samples, but coord has {}".format(
                    prediction.shape[1], coord.shape[1]
                )
            )
        return prediction
