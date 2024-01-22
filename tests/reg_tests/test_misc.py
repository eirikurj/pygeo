import os
import unittest
import numpy as np

from pygeo.geo_utils import misc

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestKSFunction(unittest.TestCase):
    def test_ksFunction(self):
        # Sequence of values
        g = np.arange(100, dtype=np.float_)
        KS = misc.ksFunction(g, rho=1e2)
        np.testing.assert_allclose(KS, np.max(g))

        # All zeros
        g = np.zeros(100)
        KS = misc.ksFunction(g, rho=1e2)
        np.testing.assert_allclose(KS, 4.605170185988092e-02)

        KS = misc.ksFunction(g, rho=1e6)
        np.testing.assert_allclose(KS, 4.605170185988092e-06)

        # Random values (with fixed seed)
        np.random.seed(123)
        g = np.random.rand(100)
        KS = misc.ksFunction(g, rho=1e2)
        np.testing.assert_allclose(KS, np.max(g), rtol=1e-2)

    def test_ksFunction_d(self):
        def finiteDifference(g, h=1e-8):
            ks = misc.ksFunction(g)
            KSFD = np.zeros_like(g)
            for i in range(g.shape[0]):
                gFD = g.copy()
                gFD[i] += h

                KSFD[i] = (misc.ksFunction(gFD) - ks) / h

            return KSFD

        def complexStep(g, h=1e-40):
            KSCS = np.zeros_like(g)
            for i in range(g.shape[0]):
                gCS = np.zeros_like(g, dtype=np.complex_)
                gCS[:] = g.copy()
                gCS[i] += 1.0j * h

                KSCS[i] = np.imag(misc.ksFunction(gCS)) / h

            return KSCS

        # Random values (with fixed seed)
        np.random.seed(123)
        g = np.random.rand(100)

        # Check reference value is good
        dKSdgFD = finiteDifference(g)
        dKSdgCS = complexStep(g)
        np.testing.assert_allclose(dKSdgCS, dKSdgFD, rtol=0, atol=1e-6)

        # Check analytic derivatives
        dKSdg = misc.ksFunction_d(g, rho=1e2)
        np.testing.assert_allclose(dKSdg, dKSdgCS)

        # Integer values
        g = np.arange(100, dtype=np.float_)
        dKSdgCS = complexStep(g)
        dKSdg = misc.ksFunction_d(g, rho=1e2)
        np.testing.assert_allclose(dKSdg, dKSdgCS, rtol=0, atol=1e-16)

        # All zeros
        g = np.zeros(100)
        dKSdgCS = complexStep(g)
        dKSdg = misc.ksFunction_d(g, rho=1e2)
        np.testing.assert_allclose(dKSdg, dKSdgCS)

        dKSdg = misc.ksFunction_d(g, rho=1e6)
        np.testing.assert_allclose(dKSdg, dKSdgCS)


if __name__ == "__main__":
    unittest.main()
