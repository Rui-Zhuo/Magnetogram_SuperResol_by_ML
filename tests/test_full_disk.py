"""Synthetic checks of geometry, patch assembly and interpolation provenance."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from full_disk import generate


class FullDiskTests(unittest.TestCase):
    def test_paper_grid(self):
        self.assertEqual(generate.centered_grid((4096,4096)), ((20,20),(48,48)))
        with self.assertRaises(ValueError):
            generate.centered_grid((200,200))

    def test_patch_placement_and_fallback_mask(self):
        field = np.full((400,400), 5., dtype=np.float32)
        field[20,20] = np.nan
        radius = lambda y,x,size: np.full((size,size), x+y, dtype=np.float32)

        def prediction(model, hmi, radius, shape, chunk_size):
            return np.full(shape, radius[0,0]+10, dtype=np.float32)

        with patch.object(generate,'predict',side_effect=prediction) as predict, contextlib.redirect_stdout(io.StringIO()):
            arrays, records, offsets = generate.reconstruct(field,radius,object())
        self.assertEqual(predict.call_count,3)
        self.assertEqual(offsets,(0,0))
        self.assertEqual(arrays['SRfield'].shape,(624,672))
        self.assertTrue(np.isnan(arrays['SRfield'][:312,:336]).all())
        self.assertTrue((arrays['SRfield'][:312,336:]==210).all())
        self.assertTrue((arrays['SRfield'][312:,:336]==210).all())
        self.assertTrue((arrays['SRfield'][312:,336:]==410).all())
        self.assertEqual(arrays['source_mask'][0,0],2)
        self.assertAlmostEqual(arrays['SRFfield'][0,0],5.)
        self.assertTrue((arrays['source_mask'][312:,336:]==1).all())
        self.assertTrue((arrays['source_mask']==0).any())
        self.assertEqual(arrays['SRfield'].dtype,np.float32)
        self.assertEqual(arrays['SRFfield'].dtype,np.float64)
        self.assertEqual(records[0]['status'],'interpolation: non-finite HMI patch')

    def test_model_failure_is_not_silently_filled(self):
        field=np.ones((400,400),dtype=np.float32)
        radius=lambda y,x,size:np.zeros((size,size),dtype=np.float32)
        with patch.object(generate,'predict',side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                generate.reconstruct(field,radius,object())


if __name__ == '__main__':
    unittest.main()
