import unittest
import numpy as np
from pygeo.geo_utils import rotation


class TestRotation(unittest.TestCase):
    def setUp(self):
        self.theta = 90.0

    def _csM(self, rotFunc):
        h = 1e-40
        thetaCS = self.theta + 1.0j * h
        MCS = np.imag(rotFunc(thetaCS)) / h
        return MCS

    def test_rotzM(self):
        M = rotation.rotzM(self.theta)
        MTrue = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(M, MTrue, atol=1e-16)

    def test_rotzM_d(self):
        Md = rotation.rotzM_d(self.theta, 1.0)
        MCS = self._csM(rotation.rotzM)
        np.testing.assert_allclose(Md, MCS)

    def test_rotzM_b(self):
        # Only do a dot product test
        thetad = np.random.rand()
        Md = rotation.rotzM_d(self.theta, thetad)
        Mb = np.random.rand(*Md.shape)
        thetab = rotation.rotzM_b(self.theta, Mb)
        np.testing.assert_allclose(np.dot(Md.flatten(), Mb.flatten()), thetad * thetab)

    def test_rotM(self):
        M = rotation.rotM(self.theta)
        MTrue = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_allclose(M, MTrue, atol=1e-16)

    def test_rotM_d(self):
        Md = rotation.rotM_d(self.theta, 1.0)
        MCS = self._csM(rotation.rotM)
        np.testing.assert_allclose(Md, MCS)

    def test_rotM_b(self):
        # Only do a dot product test
        thetad = np.random.rand()
        Md = rotation.rotM_d(self.theta, thetad)
        Mb = np.random.rand(*Md.shape)
        thetab = rotation.rotM_b(self.theta, Mb)
        np.testing.assert_allclose(np.dot(Md.flatten(), Mb.flatten()), thetad * thetab)

    def test_rotate(self):
        coords = np.array([[2.0, 0.0], [10.0, 4.0], [8.0, 8.0], [0.0, 4.0]])
        M = rotation.rotM(self.theta)
        coordsRotatedd = rotation.rotate(coords, M)
        coordsRotatedTrue = np.array([[0.0, 2.0], [-4.0, 10.0], [-8.0, 8.0], [-4.0, 0.0]])
        np.testing.assert_allclose(coordsRotatedd, coordsRotatedTrue, rtol=1e-15, atol=1e-15)

    def test_rotate_d(self):
        coords = np.array([[2.0, 0.0], [10.0, 4.0], [8.0, 8.0], [0.0, 4.0]])
        M = rotation.rotM(self.theta)

        # Use complex step as reference
        h = 1e-40
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                coordsCS = np.zeros_like(coords, dtype=np.complex_)
                coordsCS[:] = coords.copy()
                coordsCS[i, j] += 1.0j * h
                coordsRotatedCS = rotation.rotate(coordsCS, M)
                coordsRotatedCS = np.imag(coordsRotatedCS) / h

                coordsd = np.zeros_like(coords)
                coordsd[i, j] = 1.0
                Md = rotation.rotM_d(self.theta, 0.0)
                coordsRotatedd = rotation.rotate_d(coords, coordsd, M, Md)

                np.testing.assert_allclose(coordsRotatedd, coordsRotatedCS)

    def test_rotate_b(self):
        # Just test using a dot product test
        coords = np.array([[2.0, 0.0], [10.0, 4.0], [8.0, 8.0], [0.0, 4.0]])
        M = rotation.rotM(self.theta)

        # Generate random seeds
        coordsd = np.random.rand(*coords.shape)
        Md = np.random.rand(*M.shape)
        coordsRotatedd = rotation.rotate_d(coords, coordsd, M, Md)
        Mb = np.zeros_like(M)
        coordsb = np.zeros_like(coords)
        coordsRotatedb = np.random.rand(*coords.shape)
        rotation.rotate_b(coords, coordsb, M, Mb, coordsRotatedb)

        xx = np.dot(coordsd.flatten(), coordsb.flatten()) + np.dot(Md.flatten(), Mb.flatten())
        yy = np.dot(coordsRotatedd.flatten(), coordsRotatedb.flatten())
        np.testing.assert_allclose(xx, yy)


if __name__ == "__main__":
    unittest.main()
