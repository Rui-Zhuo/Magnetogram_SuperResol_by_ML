"""Physical unit checks for exported absolute fluxes and dimensionless ratios."""
import unittest

import numpy as np
from evaluation.metrics import flux_weber


class FluxUnitsTests(unittest.TestCase):
    def test_one_gauss_over_one_square_metre(self):
        flux = flux_weber(np.array([[1.]]), 1.)
        self.assertAlmostEqual(flux['net_Wb'], 1e-4)
        self.assertAlmostEqual(flux['unsigned_Wb'], 1e-4)

    def test_signed_and_unsigned_flux_for_mixed_polarities(self):
        flux = flux_weber(np.array([[3., -1.]]), 2.)
        self.assertAlmostEqual(flux['net_Wb'], 4e-4)
        self.assertAlmostEqual(flux['unsigned_Wb'], 8e-4)
        cancelled = flux_weber(np.array([[1., -1.]]), 2.)
        self.assertEqual(cancelled['net_Wb'], 0.)
        self.assertAlmostEqual(cancelled['unsigned_Wb'], 4e-4)

    def test_flux_ratios_do_not_depend_on_common_unit_conversion(self):
        reference = np.array([[4., -1.]])
        prediction = np.array([[2., -1.]])
        original = flux_weber(reference, 3.)
        reconstructed = flux_weber(prediction, 3.)
        self.assertAlmostEqual(reconstructed['net_Wb'] / original['net_Wb'], 1. / 3.)
        self.assertAlmostEqual(reconstructed['unsigned_Wb'] / original['unsigned_Wb'], 3. / 5.)


if __name__ == '__main__':
    unittest.main()
