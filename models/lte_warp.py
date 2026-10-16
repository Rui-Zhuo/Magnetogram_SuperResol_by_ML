"""Portable two-cell Fourier decoder with explicit position-modulation versions.

New models use one patch-center weight. The legacy query-minimum rule is
retained explicitly for the distributed, unmodified historical checkpoints.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from .models import register
from models.coordinates import make_coord

import numpy as np

@register('lte-warp')
class LTEWarp(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, use_pm=True,
                 position_mode='patch_center', position_input_scale=200.0,
                 solar_radius_arcsec=960.0, pm_min=0.5, pm_max=1.5):
        super().__init__()
        if position_mode not in ('patch_center', 'legacy_query_min'):
            raise ValueError('position_mode must be patch_center or legacy_query_min')
        if not all(math.isfinite(x) and x > 0 for x in
                   (position_input_scale, solar_radius_arcsec, pm_min, pm_max)) or pm_min > pm_max:
            raise ValueError('Position scales and ordered modulation bounds must be finite and positive')
        self.use_pm = use_pm
        self.position_mode = position_mode
        self.position_input_scale = float(position_input_scale)
        self.solar_radius_arcsec = float(solar_radius_arcsec)
        self.pm_min, self.pm_max = float(pm_min), float(pm_max)
        self.query_distance_min = None
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).to(inp.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        if cell is None or cell.shape[:-1] != coord.shape[:-1] or cell.shape[-1] != 2:
            raise ValueError('The portable decoder requires two cell-size features per query; '
                             'ten-feature Jacobian checkpoints require their original pipeline')
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

                inp = torch.mul(q_coef, q_freq)

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)



        grid_out = F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
            padding_mode='border', align_corners=False)[:, :, 0, :]\
                .permute(0, 2, 1)

        field = grid_out[:, :, 0:1]
        distn = grid_out[:, :, 1:2]

        if self.use_pm:
            if self.position_mode == 'patch_center':
                # Txy is normalized by 200 with HMI. Restore arcsec before
                # dividing the patch-center radius by the reference 960 arcsec.
                h_center, w_center = self.inp.shape[-2] // 2, self.inp.shape[-1] // 2
                dist_norm = (self.inp[:, 1, h_center, w_center]
                             * self.position_input_scale / self.solar_radius_arcsec).view(-1, 1, 1)
            else:
                # Exact historical rule; do not reinterpret old trained tensors.
                distance_min = distn.min() if self.query_distance_min is None else self.query_distance_min
                dist_norm = (distn - distance_min) / self.solar_radius_arcsec
            mod_factor = (1 - self.gamma * dist_norm + self.beta).clamp(self.pm_min, self.pm_max)
            field = field * mod_factor
        ret = ret + field

        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
