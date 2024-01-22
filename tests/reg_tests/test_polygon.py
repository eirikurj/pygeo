import os
import unittest
import numpy as np

from pygeo.geo_utils import polygon, rotation

baseDir = os.path.dirname(os.path.abspath(__file__))


# Functions generating coordinates


def getRTRectCoords():
    """
    Simple static coords of rotated and translated rectangle (counterclockwise coords)
    """
    return np.array([[2.0, 0.0], [10.0, 4.0], [8.0, 8.0], [0.0, 4.0]])


def computeRectCoords(b, h, xc=0.0, yc=0.0):
    """
    Generate coordinates in counterclockwise direction for a rectangle
    with base b and height h and centroid at (xc,yc)
    """
    coords = np.array(
        [[xc - b / 2, yc - h / 2], [xc + b / 2, yc - h / 2], [xc + b / 2, yc + h / 2], [xc - b / 2, yc + h / 2]]
    )

    return coords


def computeCircleCoords(r, N=50, x0=0.0, y0=0.0, thetaStart=0.0, thetaStop=2 * np.pi, endpoint=False):
    """
    Generate coordinates in counterclockwise direction"""
    coords = np.zeros((N, 2))
    theta = np.linspace(thetaStart, thetaStop, num=N, endpoint=endpoint)
    coords[:, 0] = r * np.cos(theta) + x0
    coords[:, 1] = r * np.sin(theta) + y0

    return coords


def computeParabolicSpandrelCoords(b, h, N=100):
    """
    Generate coordinates in counterclockwise direction for a parabolic spandrel
    with base b and height h. Note that this is a concave geometry.
    """
    # N-1 points are used to resolve the parabolic part and one for the corner
    x = np.linspace(b, 0, num=N - 1)
    y = (h / b**2) * x**2

    coords = np.zeros((N, 2))
    coords[0, 0] = b
    coords[0, 1] = 0.0
    coords[1:, 0] = x
    coords[1:, 1] = y

    return coords


class TestPolygon(unittest.TestCase):
    def test_areaPoly2(self):
        # ------------------------------------------------------------------
        # Square (counterclockwise coords)
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        area = polygon.areaPoly2(coords)
        np.testing.assert_allclose(area, 1.0)

        # ------------------------------------------------------------------
        # Rotated and translated rectangle (counterclockwise coords)
        coords = getRTRectCoords()

        area = polygon.areaPoly2(coords)
        np.testing.assert_allclose(area, 40.0)

        # ------------------------------------------------------------------
        # Circle at (x0,y0)=(2,3) with radius of 4
        x0 = 2.0
        y0 = 3.0
        r = 4.0
        N = 100

        coords = computeCircleCoords(r, N=N, x0=x0, y0=y0)
        area = polygon.areaPoly2(coords)
        # Analytic area value
        areaTrue = np.pi * r**2
        # Circle area is hard, so we set the relative to match at least the fourth digit for this few points
        np.testing.assert_allclose(area, areaTrue, rtol=1e-3)

        # ------------------------------------------------------------------
        # Semicircle at (x0,y0)=(2,3) with radius of 4
        coords = computeCircleCoords(r, N=N, x0=x0, y0=y0, thetaStop=np.pi, endpoint=True)
        area = polygon.areaPoly2(coords)
        areaTrue = np.pi * r**2 / 2
        np.testing.assert_allclose(area, areaTrue, rtol=1e-3)

        # ------------------------------------------------------------------
        # Concave test with parabolic spandrel with base b=5 and height h=6
        b = 5.0
        h = 6.0
        coords = computeParabolicSpandrelCoords(b, h, N=100)
        area = polygon.areaPoly2(coords)
        # Analytic area value
        areaTrue = b * h / 3
        np.testing.assert_allclose(area, areaTrue, rtol=1e-4)

    def test_areaPoly2_b(self):
        # ------------------------------------------------------------------
        # Rotated and translated rectangle (counterclockwise coords)
        coords = getRTRectCoords()

        # Compute complex step reference
        h = 1e-40
        areaCS = np.zeros_like(coords)
        N = coords.shape[0]
        for i in range(N):
            for j in range(2):
                coordsCS = np.zeros_like(coords, dtype=np.complex_)
                coordsCS[:] = coords.copy()
                coordsCS[i, j] += 1.0j * h
                areaCS[i, j] = np.imag(polygon.areaPoly2(coordsCS)) / h

        # Reverse mode
        dAreadCoords = np.zeros_like(coords)
        polygon.areaPoly2_b(coords, dAreadCoords)
        np.testing.assert_allclose(dAreadCoords, areaCS)

        # Analytic
        dAreadCoords = polygon.areaPoly2_d(coords)
        np.testing.assert_allclose(dAreadCoords, areaCS)

    def test_centroidPoly(self):
        # ------------------------------------------------------------------
        # Semicircle at (x0,y0)=(0,0) with radius of 4
        r = 4.0
        coords = computeCircleCoords(r, N=100, x0=0.0, y0=0.0, thetaStop=np.pi, endpoint=True)
        xc, yc = polygon.centroidPoly(coords)
        xcTrue = 0.0
        ycTrue = (4 * r) / (3 * np.pi)
        np.testing.assert_allclose(xc, xcTrue, rtol=0, atol=1e-10)
        np.testing.assert_allclose(yc, ycTrue, rtol=1e-4)

        # ------------------------------------------------------------------
        # Concave test with parabolic spandrel with base b=5 and height h=6
        b = 5.0
        h = 6.0
        coords = computeParabolicSpandrelCoords(b, h, N=100)
        xc, yc = polygon.centroidPoly(coords)

        # Analytic values
        xcTrue = 3 * b / 4
        ycTrue = 3 * h / 10
        np.testing.assert_allclose(xc, xcTrue, rtol=1e-4)
        np.testing.assert_allclose(yc, ycTrue, rtol=1e-4)

    def test_centroidPoly_b(self):
        # ------------------------------------------------------------------
        # Rotated and translated rectangle (counterclockwise coords)
        coords = getRTRectCoords()

        # Compute complex step reference
        h = 1e-40
        xcCS = np.zeros_like(coords)
        ycCS = np.zeros_like(coords)
        N = coords.shape[0]
        for i in range(N):
            for j in range(2):
                coordsCS = np.zeros_like(coords, dtype=np.complex_)
                coordsCS[:] = coords.copy()
                coordsCS[i, j] += 1.0j * h
                xc, yc = polygon.centroidPoly(coordsCS)
                xcCS[i, j] = np.imag(xc) / h
                ycCS[i, j] = np.imag(yc) / h

        # Reverse
        dxcdCoords = np.zeros_like(coords)
        dycdCoords = np.zeros_like(coords)
        polygon.centroidPoly_b(coords, dxcdCoords, xcb=1.0, ycb=0.0)
        polygon.centroidPoly_b(coords, dycdCoords, xcb=0.0, ycb=1.0)
        np.testing.assert_allclose(dxcdCoords, xcCS)
        np.testing.assert_allclose(dycdCoords, ycCS)

        # Analytic
        dxcdCoords, dycdCoords = polygon.centroidPoly_d(coords)
        np.testing.assert_allclose(dxcdCoords, xcCS)
        np.testing.assert_allclose(dycdCoords, ycCS)

    def test_secondMomentAreaPoly(self):
        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes and centroid coincident at (0,0)
        # Compute second moments of area at origin
        b = 2.0
        h = 4.0
        coords = computeRectCoords(b, h, xc=0.0, yc=0.0)
        # Setting aboutCentroid to True/False will give the same value as the centroid is at the origin
        # Regardless, setting it to True to make sure it computes about origin (same as analytic)
        Ixx, Ixy, Iyy, Jz = polygon.secondMomentAreaPoly(coords, aboutCentroid=True)

        # Check that same value is obtained with aboutCentroid as True or False
        np.testing.assert_allclose((Ixx, Ixy, Iyy, Jz), polygon.secondMomentAreaPoly(coords, aboutCentroid=False))

        # Analytically integrated values
        IxxTrue = b * h**3 / 12
        IxyTrue = 0.0
        IyyTrue = b**3 * h / 12
        JzTrue = IxxTrue + IyyTrue

        np.testing.assert_allclose(Ixx, IxxTrue)
        np.testing.assert_allclose(Ixy, IxyTrue)
        np.testing.assert_allclose(Iyy, IyyTrue)
        np.testing.assert_allclose(Jz, JzTrue)

        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes at left lower corner (0,0)
        # Centroid is at (1,2). Compute second moments of area at origin.
        b = 2.0
        h = 4.0
        coords = computeRectCoords(b, h, xc=b / 2, yc=h / 2)
        Ixx, Ixy, Iyy, Jz = polygon.secondMomentAreaPoly(coords, aboutCentroid=False)

        # Analytically integrated values at origin
        IxxTrue = b * h**3 / 3
        IxyTrue = b**2 * h**2 / 4
        IyyTrue = b**3 * h / 3
        JzTrue = IxxTrue + IyyTrue

        np.testing.assert_allclose(Ixx, IxxTrue)
        np.testing.assert_allclose(Ixy, IxyTrue)
        np.testing.assert_allclose(Iyy, IyyTrue)
        np.testing.assert_allclose(Jz, JzTrue)

        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes at the centroid at (1,2).
        # Compute second moments of area at centroid (which is not coincident with origin).
        # This should give same values as b=2 and height h=3, with origin of axes and centroid coincident at (0,0)
        b = 2.0
        h = 4.0
        coords = computeRectCoords(b, h, xc=b / 2, yc=h / 2)
        Ixx, Ixy, Iyy, Jz = polygon.secondMomentAreaPoly(coords, aboutCentroid=True)

        # Analytically integrated values at centroid
        IxxTrue = b * h**3 / 12
        IxyTrue = 0.0
        IyyTrue = b**3 * h / 12
        JzTrue = IxxTrue + IyyTrue

        np.testing.assert_allclose(Ixx, IxxTrue)
        np.testing.assert_allclose(Ixy, IxyTrue)
        np.testing.assert_allclose(Iyy, IyyTrue)
        np.testing.assert_allclose(Jz, JzTrue)

        # ------------------------------------------------------------------
        # Circle of radius r=4, with origin of axes at center
        # Centroid and origin are coincident, and thus setting aboutCentroid to True/False should give the same value
        x0 = 0.0
        y0 = 0.0
        r = 4.0
        coords = computeCircleCoords(r, N=200, x0=x0, y0=y0)
        Ixx, Ixy, Iyy, Jz = polygon.secondMomentAreaPoly(coords, aboutCentroid=False)

        # Check that same value is obtained with aboutCentroid as True or False.
        # Note that there are small arithmetic errors in computing the centroid, hence the adjusted tolerance.
        np.testing.assert_allclose(
            (Ixx, Ixy, Iyy, Jz), polygon.secondMomentAreaPoly(coords, aboutCentroid=True), rtol=0, atol=1e-13
        )

        # Analytically integrated values
        IxxTrue = np.pi * r**4 / 4
        IxyTrue = 0.0
        IyyTrue = IxxTrue
        JzTrue = 2 * IxxTrue

        np.testing.assert_allclose(Ixx, IxxTrue, rtol=1e-3)
        np.testing.assert_allclose(Ixy, IxyTrue, rtol=0, atol=1e-10)
        np.testing.assert_allclose(Iyy, IyyTrue, rtol=1e-3)
        np.testing.assert_allclose(Jz, JzTrue, rtol=1e-3)

        # ------------------------------------------------------------------
        # Concave test with parabolic spandrel with base b=5 and height h=6
        # Origin of axes at (0,0)
        b = 5.0
        h = 6.0
        coords = computeParabolicSpandrelCoords(b, h, N=100)
        Ixx, Ixy, Iyy, Jz = polygon.secondMomentAreaPoly(coords, aboutCentroid=False)

        # Analytically integrated values
        IxxTrue = b * h**3 / 21
        IxyTrue = b**2 * h**2 / 12
        IyyTrue = h * b**3 / 5
        JzTrue = IxxTrue + IyyTrue

        np.testing.assert_allclose(Ixx, IxxTrue, rtol=1e-4)
        np.testing.assert_allclose(Ixy, IxyTrue, rtol=1e-4)
        np.testing.assert_allclose(Iyy, IyyTrue, rtol=1e-4)
        np.testing.assert_allclose(Jz, JzTrue, rtol=1e-3)

        # ------------------------------------------------------------------
        # RAE 2822 airfoil
        coords = np.loadtxt(os.path.join(baseDir, "../../input_files/rae2822.dat"), skiprows=1)
        Ixx, Ixy, Iyy, Jz = polygon.secondMomentAreaPoly(coords, aboutCentroid=True)

        np.testing.assert_allclose(Ixx, 6.55809264052421e-05)
        np.testing.assert_allclose(Ixy, 7.339145066414142e-05)
        np.testing.assert_allclose(Iyy, 0.003808606544631391)
        np.testing.assert_allclose(Jz, 0.003874187471036633)

    def test_secondMomentAreaPoly_b(self):
        def _run(coords, aboutCentroid=True):
            # Compute complex step reference
            h = 1e-40
            IxxCS = np.zeros_like(coords)
            IxyCS = np.zeros_like(coords)
            IyyCS = np.zeros_like(coords)
            JzCS = np.zeros_like(coords)
            N = coords.shape[0]
            for i in range(N):
                for j in range(2):
                    coordsCS = np.zeros_like(coords, dtype=np.complex_)
                    coordsCS[:] = coords.copy()
                    coordsCS[i, j] += 1.0j * h

                    Ixx, _, _, _ = polygon.secondMomentAreaPoly(coordsCS, aboutCentroid=aboutCentroid)
                    IxxCS[i, j] = np.imag(Ixx) / h

                    _, Ixy, _, _ = polygon.secondMomentAreaPoly(coordsCS, aboutCentroid=aboutCentroid)
                    IxyCS[i, j] = np.imag(Ixy) / h

                    _, _, Iyy, _ = polygon.secondMomentAreaPoly(coordsCS, aboutCentroid=aboutCentroid)
                    IyyCS[i, j] = np.imag(Iyy) / h

                    _, _, _, Jz = polygon.secondMomentAreaPoly(coordsCS, aboutCentroid=aboutCentroid)
                    JzCS[i, j] = np.imag(Jz) / h

            # Compute AD derivatives
            dIxxdCoords = np.zeros_like(coords)
            dIxydCoords = np.zeros_like(coords)
            dIyydCoords = np.zeros_like(coords)
            dJzdCoords = np.zeros_like(coords)

            polygon.secondMomentAreaPoly_b(
                coords, dIxxdCoords, Ixxb=1.0, Ixyb=0.0, Iyyb=0.0, Jzb=0.0, aboutCentroid=aboutCentroid
            )
            polygon.secondMomentAreaPoly_b(
                coords, dIxydCoords, Ixxb=0.0, Ixyb=1.0, Iyyb=0.0, Jzb=0.0, aboutCentroid=aboutCentroid
            )
            polygon.secondMomentAreaPoly_b(
                coords, dIyydCoords, Ixxb=0.0, Ixyb=0.0, Iyyb=1.0, Jzb=0.0, aboutCentroid=aboutCentroid
            )
            polygon.secondMomentAreaPoly_b(
                coords, dJzdCoords, Ixxb=0.0, Ixyb=0.0, Iyyb=0.0, Jzb=1.0, aboutCentroid=aboutCentroid
            )

            np.testing.assert_allclose(dIxxdCoords, IxxCS)
            np.testing.assert_allclose(dIxydCoords, IxyCS)
            np.testing.assert_allclose(dIyydCoords, IyyCS)
            np.testing.assert_allclose(dJzdCoords, JzCS)

            dIxxdCoords, dIxydCoords, dIyydCoords, dJzdCoords = polygon.secondMomentAreaPoly_d(
                coords, aboutCentroid=aboutCentroid
            )

            np.testing.assert_allclose(dIxxdCoords, IxxCS)
            np.testing.assert_allclose(dIxydCoords, IxyCS)
            np.testing.assert_allclose(dIyydCoords, IyyCS)
            np.testing.assert_allclose(dJzdCoords, JzCS)

        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes and centroid coincident at (0,0)
        # Compute second moments of area at origin
        b = 2.0
        h = 4.0
        coords = computeRectCoords(b, h, xc=0.0, yc=0.0)
        _run(coords, aboutCentroid=True)

        # ------------------------------------------------------------------
        # Rotated and translated rectangle (counterclockwise coords)
        coords = getRTRectCoords()
        _run(coords, aboutCentroid=False)
        _run(coords, aboutCentroid=True)

    def test_principalAxesSecondMomentArea(self):
        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes at the centroid at (1,2).
        # Compute second moments of area at centroid (which is not coincident with origin).
        # This should give same values as b=2 and height h=3, with origin of axes and centroid coincident at (0,0)
        b = 2.0
        h = 4.0
        coords = computeRectCoords(b, h, xc=b / 2, yc=h / 2)
        Ixx, Ixy, Iyy, _ = polygon.secondMomentAreaPoly(coords, aboutCentroid=True)
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(Ixx, Ixy, Iyy)

        # Values should be the same and no rotation
        np.testing.assert_allclose(Ixx, Ipmax)
        np.testing.assert_allclose(Iyy, Ipmin)
        np.testing.assert_allclose(theta, 0.0)

        # ------------------------------------------------------------------
        # Now flip the dimensions, but should still get the same principal values as before, but rotation by 90 degrees
        b = 4.0
        h = 2.0
        coords = computeRectCoords(b, h, xc=b / 2, yc=h / 2)
        Ixx, Ixy, Iyy, _ = polygon.secondMomentAreaPoly(coords, aboutCentroid=True)
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(Ixx, Ixy, Iyy)
        # Values should be flipped, and  and no rotation
        np.testing.assert_allclose(Ixx, Ipmin)
        np.testing.assert_allclose(Iyy, Ipmax)
        np.testing.assert_allclose(theta, -90.0)

        # ------------------------------------------------------------------
        # Arbitrary angle rotation case (here about 0,0), use previous data
        M = rotation.rotM(45)
        coordsRotated = rotation.rotate(coords, M)

        IxxRotated, IxyRotated, IyyRotated, _ = polygon.secondMomentAreaPoly(coordsRotated, aboutCentroid=True)
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(IxxRotated, IxyRotated, IyyRotated)

        # Compare the principal values to the unrotated values from before
        np.testing.assert_allclose(Ixx, Ipmin)
        np.testing.assert_allclose(Iyy, Ipmax)
        np.testing.assert_allclose(theta, -45.0)

    def test_principalAxesSecondMomentArea_b(self):
        # ------------------------------------------------------------------
        # Simple box
        # b = 4.0
        # h = 2.0
        # coords = computeRectCoords(b, h, xc=b/2, yc=h/2)

        # b = 1.0
        # h = 1.0
        # coords = computeRectCoords(b, h, xc=b/2, yc=h/2)

        # Rotated and translated rectangle (counterclockwise coords)
        coords = getRTRectCoords()

        Ixx, Ixy, Iyy, _ = polygon.secondMomentAreaPoly(coords, aboutCentroid=True)

        # Compute complex step reference
        h = 1e-40
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(Ixx + 1.0j * h, Ixy, Iyy)
        dIpmaxdIxxCS = np.imag(Ipmax) / h
        dIpmindIxxCS = np.imag(Ipmin) / h
        dthetadIxxCS = np.imag(theta) / h
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(Ixx, Ixy + 1.0j * h, Iyy)
        dIpmaxdIxyCS = np.imag(Ipmax) / h
        dIpmindIxyCS = np.imag(Ipmin) / h
        dthetadIxyCS = np.imag(theta) / h
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(Ixx, Ixy, Iyy + 1.0j * h)
        dIpmaxdIyyCS = np.imag(Ipmax) / h
        dIpmindIyyCS = np.imag(Ipmin) / h
        dthetadIyyCS = np.imag(theta) / h

        # FD
        hFD = 1e-5
        Ipmax, Ipmin, theta = polygon.principalAxesSecondMomentArea(Ixx, Ixy, Iyy)
        Ipmaxdh, Ipmindh, thetadh = polygon.principalAxesSecondMomentArea(Ixx + hFD, Ixy, Iyy)

        dIpmaxdIxxFD = (Ipmaxdh - Ipmax) / hFD
        dIpmindIxxFD = (Ipmindh - Ipmin) / hFD
        dthetadIxxFD = (thetadh - theta) / hFD

        np.testing.assert_allclose(dIpmaxdIxxFD, dIpmaxdIxxCS)
        np.testing.assert_allclose(dIpmindIxxFD, dIpmindIxxCS)
        np.testing.assert_allclose(dthetadIxxFD, dthetadIxxCS)

        # Analytic
        (
            dIpmaxdIxx,
            dIpmaxdIxy,
            dIpmaxdIyy,
            dIpmindIxx,
            dIpmindIxy,
            dIpmindIyy,
            dthetadIxx,
            dthetadIxy,
            dthetadIyy,
        ) = polygon.principalAxesSecondMomentArea_d(Ixx, Ixy, Iyy)

        np.testing.assert_allclose(dIpmaxdIxx, dIpmaxdIxxCS)
        np.testing.assert_allclose(dIpmaxdIxy, dIpmaxdIxyCS)
        np.testing.assert_allclose(dIpmaxdIyy, dIpmaxdIyyCS)

        np.testing.assert_allclose(dIpmindIxx, dIpmindIxxCS)
        np.testing.assert_allclose(dIpmindIxy, dIpmindIxyCS)
        np.testing.assert_allclose(dIpmindIyy, dIpmindIyyCS)

        np.testing.assert_allclose(dthetadIxx, dthetadIxxCS)
        np.testing.assert_allclose(dthetadIxy, dthetadIxyCS)
        np.testing.assert_allclose(dthetadIyy, dthetadIyyCS)

        # Reverse mode
        dIpmaxdIxx = 0.0
        dIpmaxdIxy = 0.0
        dIpmaxdIyy = 0.0
        dIpmaxdIxx, dIpmaxdIxy, dIpmaxdIyy = polygon.principalAxesSecondMomentArea_b(
            Ixx, dIpmaxdIxx, Ixy, dIpmaxdIxy, Iyy, dIpmaxdIyy, Ipmaxb=1.0, Ipminb=0.0, thetab=0.0
        )

        dIpmindIxx = 0.0
        dIpmindIxy = 0.0
        dIpmindIyy = 0.0
        dIpmindIxx, dIpmindIxy, dIpmindIyy = polygon.principalAxesSecondMomentArea_b(
            Ixx, dIpmindIxx, Ixy, dIpmindIxy, Iyy, dIpmindIyy, Ipmaxb=0.0, Ipminb=1.0, thetab=0.0
        )

        dthetadIxx = 0.0
        dthetadIxy = 0.0
        dthetadIyy = 0.0
        dthetadIxx, dthetadIxy, dthetadIyy = polygon.principalAxesSecondMomentArea_b(
            Ixx, dthetadIxx, Ixy, dthetadIxy, Iyy, dthetadIyy, Ipmaxb=0.0, Ipminb=0.0, thetab=1.0
        )

        np.testing.assert_allclose(dIpmaxdIxx, dIpmaxdIxxCS)
        np.testing.assert_allclose(dIpmaxdIxy, dIpmaxdIxyCS)
        np.testing.assert_allclose(dIpmaxdIyy, dIpmaxdIyyCS)

        np.testing.assert_allclose(dIpmindIxx, dIpmindIxxCS)
        np.testing.assert_allclose(dIpmindIxy, dIpmindIxyCS)
        np.testing.assert_allclose(dIpmindIyy, dIpmindIyyCS)

        np.testing.assert_allclose(dthetadIxx, dthetadIxxCS)
        np.testing.assert_allclose(dthetadIxy, dthetadIxyCS)
        np.testing.assert_allclose(dthetadIyy, dthetadIyyCS)

    def test_sectionModulus(self):
        # Rotated and translated rectangle (counterclockwise coords)
        coords = getRTRectCoords()

        S = polygon.sectionModulus(coords, principalAxes=False)
        np.testing.assert_allclose(S, 2.665127228812929e01)

        S = polygon.sectionModulus(coords, principalAxes=True)
        np.testing.assert_allclose(S, 2.975275368217124e01)

        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes at centroid (0,0)
        b = 2.0
        h = 4.0
        coords = computeRectCoords(b, h, xc=0.0, yc=0.0)
        S = polygon.sectionModulus(coords)

        # Analytic values
        IxxTrue = b * h**3 / 12
        yMaxTrue = h / 2
        STrue = IxxTrue / yMaxTrue

        # Due to the KS function the computed value will be somewhat off, explaining the loose tolerance
        np.testing.assert_allclose(S, STrue, rtol=1e-2)

        # ------------------------------------------------------------------
        # Semicircle of radius 4 with origin of axes at centroid
        # First its generated at (x0,y0)=(0,0), then we shift axes to centroid
        r = 4.0
        N = 100
        coords = computeCircleCoords(r, N=N, x0=0.0, y0=0.0, thetaStop=np.pi, endpoint=True)

        # Analytic values
        IxxTrue = (9 * np.pi**2 - 64) * r**4 / (72 * np.pi)
        ycTrue = (4 * r) / (3 * np.pi)
        yMaxTrue = r - ycTrue
        STrue = IxxTrue / yMaxTrue

        # Shift the coords such that the origin is at the centroid (only need y)
        coords[:, 1] -= ycTrue
        S = polygon.sectionModulus(coords)
        np.testing.assert_allclose(S, STrue, rtol=1e-2)

        # ------------------------------------------------------------------
        # RAE 2822 airfoil
        coords = np.loadtxt(os.path.join(baseDir, "../../input_files/rae2822.dat"), skiprows=1)
        S = polygon.sectionModulus(coords)
        np.testing.assert_allclose(S, 9.229645979119877e-04)

        SPrincipal = polygon.sectionModulus(coords, principalAxes=True)
        SPrincipalTrue = 9.138350650016578e-04
        np.testing.assert_allclose(SPrincipal, SPrincipalTrue)

        # Test large rotation to make sure that section modulus about principal axes is still the same
        # Rotate coords 45 degrees (here about (0,0). Doing it about centroid will not change the result
        M = rotation.rotM(45)
        coordsRotated = rotation.rotate(coords, M)
        SPrincipal = polygon.sectionModulus(coordsRotated, principalAxes=True)
        np.testing.assert_allclose(SPrincipal, SPrincipalTrue)

    def test_sectionModulus_b(self):
        def _run(coords, FDCSrtol=1e-4, principalAxes=False):
            # FD reference
            h = 1e-6
            SFD = np.zeros_like(coords)
            N = coords.shape[0]
            for i in range(N):
                for j in range(2):
                    coordsFD = coords.copy()
                    coordsFD[i, j] += h
                    SFD[i, j] = (
                        polygon.sectionModulus(coordsFD, principalAxes=principalAxes)
                        - polygon.sectionModulus(coords, principalAxes=principalAxes)
                    ) / h

            # Complex step reference
            h = 1e-40
            SCS = np.zeros_like(coords)
            N = coords.shape[0]
            for i in range(N):
                for j in range(2):
                    coordsCS = np.zeros_like(coords, dtype=np.complex_)
                    coordsCS[:] = coords.copy()
                    coordsCS[i, j] += 1.0j * h

                    S = polygon.sectionModulus(coordsCS, principalAxes=principalAxes)
                    SCS[i, j] = np.imag(S) / h

            # Check CS to FD reference for complex step issues
            np.testing.assert_allclose(SCS, SFD, rtol=FDCSrtol)

            # Compute derivatives and compare to CS reference
            dSdCoords = np.zeros_like(coords)
            polygon.sectionModulus_b(coords, dSdCoords, Sb=1.0, principalAxes=principalAxes)
            np.testing.assert_allclose(dSdCoords, SCS)

        # ------------------------------------------------------------------
        # Rectangle with base b=2 and height h=3, with origin of axes at centroid (0,0)
        b = 2.0
        h = 4.0
        coords = getRTRectCoords()
        _run(coords, FDCSrtol=1e-4, principalAxes=True)
        _run(coords, FDCSrtol=1e-4)

        # ------------------------------------------------------------------
        # Semicircle of radius 4 with origin of axes at centroid (x0,y0)=(0,0)
        r = 4.0
        N = 100
        coords = computeCircleCoords(r, N=N, x0=0.0, y0=0.0, thetaStop=np.pi, endpoint=True)
        _run(coords, FDCSrtol=5e-3, principalAxes=True)
        _run(coords, FDCSrtol=5e-3)

        # ------------------------------------------------------------------
        # RAE 2822 airfoil
        coords = np.loadtxt(os.path.join(baseDir, "../../input_files/rae2822.dat"), skiprows=1)
        _run(coords, FDCSrtol=6e-3, principalAxes=True)
        _run(coords, FDCSrtol=5e-2)


if __name__ == "__main__":
    unittest.main()
