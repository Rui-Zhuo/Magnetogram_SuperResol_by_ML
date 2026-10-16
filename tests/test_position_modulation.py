"""Analytic PM, checkpoint-compatibility and query-chunk regressions."""
import copy
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch
from torch import nn

import models
from inference.predict import load_model, predict
from models.coordinates import make_coord
from training.train import validate_resume_spec, version_training_spec


@models.register('_test_pm_identity_encoder')
class IdentityEncoder(nn.Module):
    out_dim = 2

    def forward(self, inp):
        return inp


def model_spec(mode='patch_center', use_pm=True):
    return dict(name='lte-warp', args=dict(
        encoder_spec=dict(name='_test_pm_identity_encoder', args={}),
        imnet_spec=dict(name='mlp', args=dict(out_dim=1, hidden_list=[4])),
        hidden_dim=4, use_pm=use_pm, position_mode=mode))


def skip_only_model(mode='patch_center', use_pm=True, gamma=.4, beta=.1):
    model = models.make(model_spec(mode, use_pm)).eval()
    with torch.no_grad():
        for parameter in model.imnet.parameters():
            parameter.zero_()
        model.gamma.fill_(gamma)
        model.beta.fill_(beta)
    return model


class PositionModulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_patch_center_restores_arcsec_and_is_uniform_per_patch(self):
        model = skip_only_model()
        # Query radii vary widely; only each input patch's center controls PM.
        inp = torch.zeros(2, 2, 4, 4)
        inp[:, 0] = 10 / 200
        inp[:, 1] = torch.arange(16).reshape(4, 4) / 200
        inp[0, 1, 2, 2], inp[1, 1, 2, 2] = 480 / 200, 960 / 200
        coord = make_coord((5, 7))[None].expand(2, -1, -1)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / 5
        cell[..., 1] *= 2 / 7
        with torch.inference_mode():
            result = model(inp, coord, cell) * 200
        # clip(1-.4*.5+.1)=.9; clip(1-.4*1+.1)=.7.
        torch.testing.assert_close(result[0], torch.full_like(result[0], 9.))
        torch.testing.assert_close(result[1], torch.full_like(result[1], 7.))

    def test_patch_center_does_not_depend_on_other_batch_members(self):
        model = skip_only_model()
        inp = torch.zeros(2, 2, 4, 4)
        inp[:, 0] = .05
        inp[0, 1] = 480 / 200
        inp[1, 1] = 0
        coord = make_coord((3, 3))[None].expand(2, -1, -1)
        cell = torch.ones_like(coord) * 2 / 3
        with torch.inference_mode():
            together = model(inp, coord, cell)[0]
            alone = model(inp[:1], coord[:1], cell[:1])[0]
        torch.testing.assert_close(together, alone)

    def test_modulation_clipping_bounds(self):
        hmi = np.full((4, 4), 10., dtype=np.float32)
        radius = np.full((4, 4), 960., dtype=np.float32)
        for gamma, expected in ((4., 5.), (-4., 15.)):
            with self.subTest(gamma=gamma):
                result = predict(skip_only_model(gamma=gamma, beta=0), hmi, radius, (5, 7), 3)
                np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_no_pm_ablation_retains_unmodulated_skip(self):
        hmi = np.full((4, 4), 10., dtype=np.float32)
        radius = np.full((4, 4), 960., dtype=np.float32)
        result = predict(skip_only_model(use_pm=False, gamma=100., beta=-100.),
                         hmi, radius, (5, 7), 4)
        np.testing.assert_allclose(result, 10., atol=1e-5)

    def test_chunking_does_not_change_either_pm_version(self):
        hmi = np.arange(16, dtype=np.float32).reshape(4, 4) * 3
        radius = 480 + np.arange(16, dtype=np.float32).reshape(4, 4) * 20
        for mode in ('patch_center', 'legacy_query_min'):
            for use_pm in (True, False):
                with self.subTest(mode=mode, use_pm=use_pm):
                    torch.manual_seed(42)
                    model = models.make(model_spec(mode, use_pm)).eval()
                    unchunked = predict(model, hmi, radius, (6, 8), 1000)
                    for chunk in (1, 7):
                        chunked = predict(model, hmi, radius, (6, 8), chunk)
                        np.testing.assert_allclose(chunked, unchunked, rtol=2e-6, atol=2e-5)

    def test_unversioned_weights_load_explicitly_as_legacy_without_mutation(self):
        model = skip_only_model()
        spec = model_spec()
        del spec['args']['position_mode']
        spec['sd'] = copy.deepcopy(model.state_dict())
        before = copy.deepcopy(spec)
        with self.assertWarnsRegex(UserWarning, 'legacy_query_min'):
            restored = models.make(spec, load_sd=True)
        self.assertEqual(restored.position_mode, 'legacy_query_min')
        self.assertNotIn('position_mode', spec['args'])
        for key in before['sd']:
            torch.testing.assert_close(restored.state_dict()[key], before['sd'][key], rtol=0, atol=0)

    def test_versioned_center_checkpoint_has_no_legacy_fallback(self):
        spec = model_spec()
        spec['sd'] = skip_only_model().state_dict()
        with warnings.catch_warnings(record=True) as captured:
            restored = models.make(spec, load_sd=True)
        self.assertEqual(restored.position_mode, 'patch_center')
        self.assertFalse(captured)

    def test_explicit_center_override_warns_and_changes_old_checkpoint_predictions(self):
        spec = model_spec()
        del spec['args']['position_mode']
        spec['sd'] = skip_only_model().state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'legacy.pth'
            torch.save(dict(model=spec), path)
            with self.assertWarnsRegex(UserWarning, 'legacy_query_min'):
                legacy = load_model(path)
            with self.assertWarnsRegex(UserWarning, 'changes predictions'):
                center = load_model(path, position_mode='patch_center')
        hmi = np.full((4, 4), 10., dtype=np.float32)
        radius = np.full((4, 4), 480., dtype=np.float32)
        np.testing.assert_allclose(predict(legacy, hmi, radius, (3, 3)), 11., atol=1e-5)
        np.testing.assert_allclose(predict(center, hmi, radius, (3, 3)), 9., atol=1e-5)

    def test_phase_weights_are_rejected_instead_of_reshaped(self):
        spec = model_spec()
        spec['sd'] = skip_only_model().state_dict()
        spec['sd']['phase.weight'] = torch.zeros(2, 10)
        with self.assertRaisesRegex(ValueError, 'not reshaped or padded'):
            models.make(spec, load_sd=True)

    def test_legacy_training_cannot_silently_resume_as_center_pm(self):
        requested = model_spec()
        saved = model_spec()
        del saved['args']['position_mode']
        with self.assertRaisesRegex(ValueError, 'Resume checkpoint PM differs'):
            validate_resume_spec(requested, saved)
        requested['args']['position_mode'] = 'legacy_query_min'
        validate_resume_spec(requested, saved)

    def test_new_training_records_center_default_for_checkpoint_roundtrip(self):
        unversioned = model_spec()
        del unversioned['args']['position_mode']
        spec = version_training_spec(unversioned)
        self.assertNotIn('position_mode', unversioned['args'])
        spec['sd'] = skip_only_model().state_dict()
        with warnings.catch_warnings(record=True) as captured:
            restored = models.make(spec, load_sd=True)
        self.assertEqual(restored.position_mode, 'patch_center')
        self.assertFalse(captured)

    def test_invalid_position_configuration_is_rejected(self):
        for override in (dict(position_mode='unknown'), dict(position_input_scale=0),
                         dict(solar_radius_arcsec=float('nan')), dict(pm_min=2., pm_max=1.)):
            with self.subTest(override=override):
                spec = model_spec()
                spec['args'].update(override)
                with self.assertRaises(ValueError):
                    models.make(spec)


if __name__ == '__main__':
    unittest.main()
