# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from .baseConstraint import GeometricConstraint
from pygeo.geo_utils import polygon


class SectionModulusConstraint(GeometricConstraint):
    """
    This class is used to represent individual section modulus constraint.
    The parameter list is explained in the addSectionModulusConstraint() of
    the DVConstraints class
    """

    def __init__(self, name, nSpan, nChord, coords, principalAxes, lower, upper, scaled, scale, DVGeo, addToPyOpt):
        super().__init__(name, nSpan, lower, upper, scale, DVGeo, addToPyOpt)

        self.nSpan = nSpan
        self.nChord = nChord
        self.coords = coords
        self.principalAxes = principalAxes
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided.
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now compute the reference section modulus
        self.S0 = self.evalSectionModulus()

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates
        self.coords = self.DVGeo.update(self.name, config=config)

        # Evaluate and scale if needed before returning
        S = self.evalSectionModulus()
        if self.scaled:
            S = np.divide(S, self.S0)

        funcs[self.name] = S

    def evalSectionModulus(self):
        """
        Evaluate the section modulus for each section at all spanwise locations
        """
        # Initialize the return quantities
        S = np.zeros(self.nSpan)

        # Reshape coordinates for convenience
        coords = self.coords.reshape((self.nSpan, self.nChord, 2, 3))

        # Loop over each spanwise location
        for i in range(self.nSpan):
            reorderedSectionNodes = self._reorderSectionCoordsCCW(coords[i, :, :, :])
            S[i] = polygon.sectionModulus(reorderedSectionNodes[:, 0:2], principalAxes=self.principalAxes)

        return S

    def _reorderSectionCoordsCCW(self, sectionCoords):
        """Reorder a section in a counterclockwise order

        Parameters
        ----------
        sectionCoords : ndarray (nChord, 2, 3)
            Input array of coordinates of dimension (nChord, 2, 3).
            The 2nd dimension (2) is the bottom and top x,y,z (3) coordinates

        Returns
        -------
        reorderedSectionCoords : ndarray (nChord*2, 3)
            Stacked coordinates with the latter half given in reverse order, resulting
            in a counterclockwise order.
        """

        reorderedSectionCoords = np.zeros((self.nChord * 2, 3))
        reorderedSectionCoords[: self.nChord, :] = sectionCoords[:, 1, :]  # Bottom x,y,z coords
        reorderedSectionCoords[self.nChord :, :] = sectionCoords[:, 0, :][::-1]  # Top x,y,z coords reversed

        return reorderedSectionCoords

    def _undoReorderSectionCoordsCCW(self, reorderedSectionCoords):
        """Inverse of _reorderSectionCoordsCCW

        Parameters
        ----------
        reorderedSectionCoords : ndarray (nChord*2, 3)
            Stacked coordinates with the latter half given in reverse order, resulting
            in a counterclockwise order.

        Returns
        -------
        sectionCoords : ndarray (nChord, 2, 3)
            Input array of coordinates of dimension (nChord, 2, 3).
            The 2nd dimension (2) is the bottom and top x,y,z (3) coordinates
        """

        sectionCoords = np.zeros((self.nChord, 2, 3))
        sectionCoords[:, 1, :] = reorderedSectionCoords[: self.nChord, :]  # Bottom x,y,z coords
        sectionCoords[:, 0, :] = reorderedSectionCoords[self.nChord :, :][::-1]  # Top x,y,z coords reversed

        return sectionCoords

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            # Compute the Jacobian
            dSdPt = self.evalSectionModulusSens()
            if self.scaled:
                for i in range(self.nSpan):
                    dSdPt[i, :, :] /= self.S0[i]

            # Now compute the DVGeo total sensitivity
            funcsSens[f"{self.name}"] = self.DVGeo.totalSensitivity(dSdPt, self.name, config=config)

    def evalSectionModulusSens(self):
        """
        Evaluate the derivative of the section modulus with respect to the coordinates.
        """

        # Reshape coordinates
        coords = self.coords.reshape((self.nSpan, self.nChord, 2, 3))

        # Allocate and do all span locations at once
        coordsb = np.zeros((self.nSpan, self.coords.shape[0], self.coords.shape[1]))

        # Loop over each spanwise location
        for i in range(self.nSpan):
            reorderedSectionNodes = self._reorderSectionCoordsCCW(coords[i, :, :, :])
            reorderedTempb = np.zeros_like(reorderedSectionNodes)
            polygon.sectionModulus_b(
                reorderedSectionNodes[:, 0:2], reorderedTempb[:, 0:2], Sb=1.0, principalAxes=self.principalAxes
            )

            # Undo the coord ordering
            tempb = np.zeros_like(coords)
            tempb[i, :, :, :] = self._undoReorderSectionCoordsCCW(reorderedTempb)

            # Reshape back to flattened array for DVGeo
            coordsb[i, :, :] = tempb.reshape((self.nSpan * self.nChord * 2, 3))

        return coordsb

    def writeTecplot(self, handle):
        """
        Writes the projected coordinates that are used to compute the geometric properties for slices
        One zone is written per slice.
        """

        coords = self.coords.reshape([self.nSpan, self.nChord, 2, 3])
        nNodesPerSection = self.nChord * 2
        nElements = nNodesPerSection

        for i in range(self.nSpan):
            handle.write(f"Zone T={self.name}_span{i}\n")
            handle.write(f"Nodes={nNodesPerSection}, Elements={nElements}, ZONETYPE=FELINESEG\n")
            handle.write("DATAPACKING=POINT\n")

            ccwCoords = self._reorderSectionCoordsCCW(coords[i, :, :, :])

            for j in range(nNodesPerSection):
                # Write the bottom section
                handle.write(f"{ccwCoords[j, 0]:f} {ccwCoords[j, 1]:f} {ccwCoords[j, 2]:f}\n")

            for j in range(nNodesPerSection):
                idx1 = j + 1
                idx2 = j + 2
                # If we have reached the end we need to close the curve
                if idx1 == nNodesPerSection:
                    idx2 = 1
                handle.write(f"{idx1} {idx2}\n")
