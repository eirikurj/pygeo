# External modules
import numpy as np

# Local modules
from . import euclideanNorm
from pygeo.geo_utils import rotation
from pygeo.geo_utils import cs
from pygeo.geo_utils.misc import ksFunction, ksFunction_d

# --------------------------------------------------------------
#                Polygon geometric functions
# --------------------------------------------------------------


def areaTri(p0, p1, p2):
    """
    Compute area based on three point arrays
    """
    # convert p1 and p2 to v1 and v2
    v1 = p1 - p0
    v2 = p2 - p0

    # compute the areas
    areaVec = np.cross(v1, v2)

    # area = np.linalg.norm(areaVec,axis=1)
    area = 0
    for i in range(len(areaVec)):
        area += euclideanNorm(areaVec[i, :])

    # return np.sum(area)/2.0
    return area / 2.0


def areaPoly(nodes):
    """Return the area of the polygon.
    Note that the input need not be strictly a polygon (closed curve in 2 dimensions).
    The approach we take here is to find the centroid, then sum the area of the
    3d triangles.

    .. warning:: This approach only works for convex polygons
    """

    c = np.average(nodes, axis=0)
    area = 0.0
    for ii in range(len(nodes)):
        xi = nodes[ii]
        xip1 = nodes[np.mod(ii + 1, len(nodes))]
        area = area + 0.5 * np.linalg.norm(np.cross(xi - c, xip1 - c))

    return np.abs(area)


def areaPoly2(nodes):
    """
    Computes the area of a polygon given a list of (x,y) coordinates in a counterclockwise order.

    No restriction is placed on the polygon shape. It can be convex or concave.
    Note that the input need not be strictly a polygon (closed curve in 2 dimensions).

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order

    Returns
    -------
    area : float
        The area of the polygon
    """

    N = nodes.shape[0]
    area = 0.0
    for i in range(N):
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0
        # A = x_i * y_(i+1) - x_(i+1)*y_i
        area += nodes[i, 0] * nodes[ip1, 1] - nodes[ip1, 0] * nodes[i, 1]

    # Normalize
    area /= 2.0

    return area


def areaPoly2_d(nodes):
    """
    Computes analytic derivative of the area w.r.t. the polygon nodes.

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order

    Returns
    -------
    dAdNodes : ndarray (N x 2)
        The derivative of the polygon area w.r.t. the nodes of the polygon
    """

    N = nodes.shape[0]
    dAdNodes = np.zeros((N, 2))
    for i in range(N):
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

        # Now compute grad(A)
        dAdNodes[i, 0] += nodes[ip1, 1]
        dAdNodes[ip1, 0] += -nodes[i, 1]
        dAdNodes[i, 1] += -nodes[ip1, 0]
        dAdNodes[ip1, 1] += nodes[i, 0]

    # Normalize
    dAdNodes /= 2.0

    return dAdNodes


def areaPoly2_b(nodes, nodesb):
    """
    Computes reverse mode derivative of the area w.r.t. the polygon nodes.

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order
    nodesb : ndarray (N x 2)
        The derivative of the polygon area w.r.t. the nodes of the polygon
    """
    # Initialize values and stack for intermediate values
    stack = []
    ip1 = 0

    N = nodes.shape[0]
    for i in range(N):
        stack.append(ip1)
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

    # Compute the reverse derivative
    # TODO: Make the seed optional
    areab = 1.0
    areab = areab / 2.0
    for i in range(N - 1, -1, -1):
        nodesb[i, 0] = nodesb[i, 0] + nodes[ip1, 1] * areab
        nodesb[ip1, 1] = nodesb[ip1, 1] + nodes[i, 0] * areab
        nodesb[ip1, 0] = nodesb[ip1, 0] - nodes[i, 1] * areab
        nodesb[i, 1] = nodesb[i, 1] - nodes[ip1, 0] * areab
        ip1 = stack.pop()


def centroidPoly(nodes):
    """
    Computes the centroid of a polygon given a list of (x,y) coordinates in a counterclockwise order.

    No restriction is placed on the polygon shape. It can be convex or concave.
    Note that the input need not be strictly a polygon (closed curve in 2 dimensions).

    The algorithm first computes the first moment of area (m^3) and then normalizes by the area.

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order

    Returns
    -------
    xc, yc : float
        The centroid of the polygon
    """

    N = nodes.shape[0]
    area = 0.0
    xc = 0.0
    yc = 0.0
    for i in range(N):
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

        # Define variables for readability
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        xip1 = nodes[ip1, 0]
        yip1 = nodes[ip1, 1]

        # compute the area
        areaTri = xi * yip1 - xip1 * yi

        area += areaTri
        xc += areaTri * (xip1 + xi)
        yc += areaTri * (yip1 + yi)

    # Normalize
    area /= 2.0
    xc /= 6.0 * area
    yc /= 6.0 * area

    return xc, yc


def centroidPoly_d(nodes):
    """
    This routine computes analytic derivative of the centroid w.r.t. the polygon nodes.

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order

    Returns
    -------
    dxcdNodes, dycdNodes : ndarray (N x 2)
        The derivatives of the polygon centroid w.r.t. the nodes
    """
    N = nodes.shape[0]
    area = 0.0

    dxcdNodes = np.zeros((N, 2))
    dycdNodes = np.zeros((N, 2))

    for i in range(N):
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

        # Define variables for readability
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        xip1 = nodes[ip1, 0]
        yip1 = nodes[ip1, 1]

        # compute the area
        areaTri = xi * yip1 - xip1 * yi
        area += areaTri

        tmpX = xip1 + xi
        tmpY = yip1 + yi

        # dxcdNodes
        dxcdNodes[i, 0] += yip1 * tmpX + areaTri
        dxcdNodes[ip1, 0] += -yi * tmpX + areaTri
        dxcdNodes[i, 1] += -xip1 * tmpX
        dxcdNodes[ip1, 1] += xi * tmpX

        # dycdNodes
        dycdNodes[i, 0] += yip1 * tmpY
        dycdNodes[ip1, 0] += -yi * tmpY
        dycdNodes[i, 1] += -xip1 * tmpY + areaTri
        dycdNodes[ip1, 1] += xi * tmpY + areaTri

    # Normalize
    area /= 2.0

    xc, yc = centroidPoly(nodes)
    dAdNodes = areaPoly2_d(nodes)
    dxcdNodes = (dxcdNodes * area - 6.0 * area * xc * dAdNodes) / (6.0 * area**2.0)
    dycdNodes = (dycdNodes * area - 6.0 * area * yc * dAdNodes) / (6.0 * area**2.0)

    return dxcdNodes, dycdNodes


def centroidPoly_b(nodes, nodesb, xcb, ycb):
    """
    Computes reverse derivatives of the centroid w.r.t. the polygon nodes.

    Typically one seed is specified at a time, unless this routine is part of a larger
    chain (remember to zero the seeds as needed).

    Example
    -------
    To compute dxc/dNodes for example, call
        centroidPoly_b(nodes, nodesb, xcb=1.0, ycb=0.0)
    """

    # Initialize values and stack for intermediate values
    stack = []
    ip1 = 0

    # Initialize output
    xc = 0.0
    yc = 0.0
    # Initialize working
    area = 0.0

    N = nodes.shape[0]
    for i in range(N):
        stack.append(ip1)
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

        # Define variables for readability
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        xip1 = nodes[ip1, 0]
        yip1 = nodes[ip1, 1]

        # compute the area
        areatri = xi * yip1 - xip1 * yi
        area = area + areatri

        # compute the centroid
        xc = xc + areatri * (xip1 + xi)
        yc = yc + areatri * (yip1 + yi)

    # Normalize
    area = area / 2.0
    tempb = ycb / (6.0 * area)
    ycb = tempb
    areab = -(yc * tempb / area)
    tempb = xcb / (6.0 * area)
    xcb = tempb
    areab = areab - xc * tempb / area
    areab = areab / 2.0

    for i in range(N - 1, -1, -1):
        xi = nodes[i, 0]
        yip1 = nodes[ip1, 1]
        xip1 = nodes[ip1, 0]
        yi = nodes[i, 1]
        areatri = xi * yip1 - xip1 * yi
        areatrib = (yip1 + yi) * ycb + (xip1 + xi) * xcb + areab
        yip1b = areatri * ycb + xi * areatrib
        yib = areatri * ycb - xip1 * areatrib
        xip1b = areatri * xcb - yi * areatrib
        xib = areatri * xcb + yip1 * areatrib
        nodesb[ip1, 1] = nodesb[ip1, 1] + yip1b
        nodesb[ip1, 0] = nodesb[ip1, 0] + xip1b
        nodesb[i, 1] = nodesb[i, 1] + yib
        nodesb[i, 0] = nodesb[i, 0] + xib
        ip1 = stack.pop()


def secondMomentAreaPoly(nodes, aboutCentroid=True):
    """
    Computes the second moments of area of a polygon
    given a list of (x,y) coordinates in a counterclockwise order.

              /                   /                   /
        Ixx = | y^2 dxdy,   Iyy = | x^2 dxdy,   Ixy = | yx dxdy
              /                   /                   /

    By default, moments are computed about the centroid of polygon, but central=False
    will use the origin of axes.

              /                        /                        /
        Ixx = | (y-yc)^2 dxdy,   Iyy = | (x-xc)^2 dxdy,   Ixy = | (y-yc)(x-xc) dxdy
              /                        /                        /

    Additionally, the polar moment of area is computed using the perpendicular axis theorem.

        Jz = Ixx + Iyy

    No restriction is placed on the polygon shape. It can be convex or concave.
    Note that the input need not be strictly a polygon (closed curve in 2 dimensions).

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order
    centroid: bool
        Compute the moments about the centroid instead about the origin


    Returns
    -------
    Ixx, Ixy, Iyy, Jz : float
        The second moment of area, and polar moment of area of the polygon
    """
    N = nodes.shape[0]
    Ixx = 0.0
    Ixy = 0.0
    Iyy = 0.0

    # Make a copy so we are not changing the input
    nodesTmp = np.copy(nodes)
    if aboutCentroid:
        # Shift to the centroid
        xc, yc = centroidPoly(nodes)
        nodesTmp[:, 0] -= xc
        nodesTmp[:, 1] -= yc

    for i in range(N):
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

        # Define variables for readability
        xi = nodesTmp[i, 0]
        yi = nodesTmp[i, 1]
        xip1 = nodesTmp[ip1, 0]
        yip1 = nodesTmp[ip1, 1]

        # Compute the area
        areaTri = xi * yip1 - xip1 * yi

        # Second moment of area
        Ixx += areaTri * (yip1**2 + yip1 * yi + yi**2)
        Ixy += areaTri * (2 * xip1 * yip1 + xip1 * yi + xi * yip1 + 2 * xi * yi)
        Iyy += areaTri * (xip1**2 + xip1 * xi + xi**2)

    # Normalize
    Ixx /= 12.0
    Ixy /= 24.0
    Iyy /= 12.0

    # Finally, compute the polar moment of area
    Jz = Ixx + Iyy

    return Ixx, Ixy, Iyy, Jz


def secondMomentAreaPoly_d(nodes, aboutCentroid=True):
    """
    This routine computes analytic derivative of Ixx, Ixy, Iyy, Jz w.r.t. the polygon nodes.

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order

    Returns
    -------
    dIxxdNodes, dIxydNodes, dIyydNodes, dJzdNodes : ndarray (N x 2)
        The derivative of the second moment of area, and polar moment of area
        w.r.t. the nodes of the polygon
    """

    # Make a copy so we are not changing the input
    nodesTmp = np.copy(nodes)
    if aboutCentroid:
        # Shift to the centroid
        xc, yc = centroidPoly(nodes)
        nodesTmp[:, 0] -= xc
        nodesTmp[:, 1] -= yc

    # Initialize outputs
    N = nodes.shape[0]
    dIxxdNodesTmp = np.zeros((N, 2))
    dIxydNodesTmp = np.zeros((N, 2))
    dIyydNodesTmp = np.zeros((N, 2))
    dJzdNodesTmp = np.zeros((N, 2))
    for i in range(N):
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

        # Define variables for readability
        xi = nodesTmp[i, 0]
        yi = nodesTmp[i, 1]
        xip1 = nodesTmp[ip1, 0]
        yip1 = nodesTmp[ip1, 1]

        # compute the area and intermediate variables
        areaTri = xi * yip1 - xip1 * yi
        tmpX = xi**2 + xi * xip1 + xip1**2
        tmpY = yi**2 + yi * yip1 + yip1**2
        tmpXY = 2 * xip1 * yip1 + xip1 * yi + xi * yip1 + 2 * xi * yi

        # Here we do dIxx/dxi, dIxx/dxip1, dIxx/yi, dIxx/dyip1
        # dIxxdNodes
        dIxxdNodesTmp[i, 0] += yip1 * tmpY
        dIxxdNodesTmp[ip1, 0] += -yi * tmpY
        dIxxdNodesTmp[i, 1] += -xip1 * tmpY + areaTri * (2 * yi + yip1)
        dIxxdNodesTmp[ip1, 1] += xi * tmpY + areaTri * (2 * yip1 + yi)

        # dIxydNodes
        dIxydNodesTmp[i, 0] += yip1 * tmpXY + areaTri * (yip1 + 2 * yi)
        dIxydNodesTmp[ip1, 0] += -yi * tmpXY + areaTri * (2 * yip1 + yi)
        dIxydNodesTmp[i, 1] += -xip1 * tmpXY + areaTri * (2 * xi + xip1)
        dIxydNodesTmp[ip1, 1] += xi * tmpXY + areaTri * (xi + 2 * xip1)

        # dIyydNodes
        dIyydNodesTmp[i, 0] += yip1 * tmpX + areaTri * (2 * xi + xip1)
        dIyydNodesTmp[ip1, 0] += -yi * tmpX + areaTri * (2 * xip1 + xi)
        dIyydNodesTmp[i, 1] += -xip1 * tmpX
        dIyydNodesTmp[ip1, 1] += xi * tmpX

    # Normalize
    dIxxdNodesTmp /= 12.0
    dIxydNodesTmp /= 24.0
    dIyydNodesTmp /= 12.0

    # dJzdNodes
    dJzdNodesTmp = dIxxdNodesTmp + dIyydNodesTmp

    # Account for the shift
    dxcdNodes, dycdNodes = centroidPoly_d(nodes)
    dNodesdNodesTmp = np.zeros((N, 2))  # Should be NxN but all but diagonal is zero
    dNodesdNodesTmp[:, 0] = 1  # - dxcdNodes[:,0] + 1 - dycdNodes[:,0]
    dNodesdNodesTmp[:, 1] = 1  # - dxcdNodes[:,1] + 1 - dycdNodes[:,1]

    # Do elementwise op instead of matrix mult
    dIxxdNodes = dIxxdNodesTmp * dNodesdNodesTmp
    dIxydNodes = dIxydNodesTmp * dNodesdNodesTmp
    dIyydNodes = dIyydNodesTmp * dNodesdNodesTmp
    dJzdNodes = dJzdNodesTmp * dNodesdNodesTmp

    return dIxxdNodes, dIxydNodes, dIyydNodes, dJzdNodes


def secondMomentAreaPoly_b(nodes, nodesb, Ixxb, Ixyb, Iyyb, Jzb, aboutCentroid=True):
    """
    Computes efficiently derivatives of Ixx, Ixy, Iyy, Jz w.r.t. the polygon nodes.

    Typically one seed is specified at a time, unless this routine is part of a larger
    chain (remember to zero the seeds as needed).

    Example
    -------
    To compute dIxx/dNodes for example,
        secondMomentAreaPoly_b(nodes, nodesb, Ixxb=1.0, Ixyb=0.0, Iyyb=0.0, Jzb=0.0)
    """

    # Initialize values and stack for intermediate values
    stack = []
    ip1 = 0

    N = nodes.shape[0]
    for i in range(N):
        stack.append(ip1)
        ip1 = i + 1
        # If at the end, circle back to the first to close the curve
        if ip1 == N:
            ip1 = 0

    # Make a copy so we are not changing the input
    nodesTmp = np.copy(nodes)
    if aboutCentroid:
        # Shift to the centroid
        xc, yc = centroidPoly(nodes)
        nodesTmp[:, 0] -= xc
        nodesTmp[:, 1] -= yc

    # Compute the reverse derivative
    Ixxb = Ixxb + Jzb
    Iyyb = Iyyb + Jzb
    Iyyb = Iyyb / 12.0
    Ixyb = Ixyb / 24.0
    Ixxb = Ixxb / 12.0
    nodesTmpb = np.zeros_like(nodesb)

    for i in range(N - 1, -1, -1):
        xi = nodesTmp[i, 0]
        yip1 = nodesTmp[ip1, 1]
        xip1 = nodesTmp[ip1, 0]
        yi = nodesTmp[i, 1]
        areatri = xi * yip1 - xip1 * yi
        areatrib = (
            (xip1**2 + xip1 * xi + xi**2) * Iyyb
            + (2 * xip1 * yip1 + xip1 * yi + xi * yip1 + 2 * xi * yi) * Ixyb
            + (yip1**2 + yip1 * yi + yi**2) * Ixxb
        )
        tempb = areatri * Iyyb
        xip1b = (2 * xip1 + xi) * tempb
        xib = (xip1 + 2 * xi) * tempb
        tempb = areatri * Ixyb
        xip1b = xip1b + (2 * yip1 + yi) * tempb - yi * areatrib
        yip1b = (2 * xip1 + xi) * tempb
        yib = (xip1 + 2 * xi) * tempb
        xib = xib + (yip1 + 2 * yi) * tempb + yip1 * areatrib
        tempb = areatri * Ixxb
        yip1b = yip1b + (2 * yip1 + yi) * tempb + xi * areatrib
        yib = yib + (yip1 + 2 * yi) * tempb - xip1 * areatrib
        nodesTmpb[ip1, 1] = nodesTmpb[ip1, 1] + yip1b
        nodesTmpb[ip1, 0] = nodesTmpb[ip1, 0] + xip1b
        nodesTmpb[i, 1] = nodesTmpb[i, 1] + yib
        nodesTmpb[i, 0] = nodesTmpb[i, 0] + xib
        ip1 = stack.pop()

    # Finally account for the shift
    if aboutCentroid:
        xcb = -np.sum(nodesTmpb[:, 0])
        ycb = -np.sum(nodesTmpb[:, 1])
        centroidPoly_b(nodes, nodesb, xcb=xcb, ycb=ycb)

    nodesb += nodesTmpb


def principalAxesSecondMomentArea(Ixx, Ixy, Iyy):
    """
    Computes the principal second moment of area and the angle of the principal axes

    Parameters
    ----------
    Ixx, Ixy, Iyy : float
        Second moments of area for a body

    Returns
    -------
    Ipmax, Ipmin : float
        The principal (maximum and minimum) second moment of area
    theta : float
        The angle (degrees) of the principal axes (maximum)

    """
    # 2*theta is computed, but we cast it to deg.
    theta2 = cs.arctan2(-2 * Ixy, Ixx - Iyy)
    theta = theta2 / 2.0 * (180 / np.pi)

    # Compute the principal second moment of area
    Iavg = (Ixx + Iyy) / 2.0
    hypotenuse = np.sqrt(((Ixx - Iyy) / 2.0) ** 2 + Ixy**2)

    Ipmax = Iavg + hypotenuse
    Ipmin = Iavg - hypotenuse

    return Ipmax, Ipmin, theta


def principalAxesSecondMomentArea_d(Ixx, Ixy, Iyy):
    """
    Computes the analytic derivatives of principal second moment of area and the angle of the principal axes w.r.t to the second moment of area

    Parameters
    ----------
    Ixx, Ixy, Iyy : float
        Second moments of area for a body

    Returns
    -------
    Ipmax, Ipmin : float
        The principal (maximum and minimum) second moment of area
    theta : float
        The angle (degrees) of the principal axes
    """

    # Compute the principal second moment of area derivatives
    IDiffHalf = (Ixx - Iyy) / 2.0
    tmp = IDiffHalf**2 + Ixy**2
    dhypotenusedIii = 0.0
    # Check for divide by zero
    if tmp != 0.0:
        dhypotenusedIii = 1 / np.sqrt(tmp)

    dIpmaxdIxx = 0.5 + 0.5 * dhypotenusedIii * IDiffHalf
    dIpmaxdIxy = 0.5 * dhypotenusedIii * 2 * Ixy
    dIpmaxdIyy = 0.5 - 0.5 * dhypotenusedIii * IDiffHalf

    dIpmindIxx = 0.5 - 0.5 * dhypotenusedIii * IDiffHalf
    dIpmindIxy = -0.5 * dhypotenusedIii * 2 * Ixy
    dIpmindIyy = 0.5 + 0.5 * dhypotenusedIii * IDiffHalf

    dthetadIxx = 0.0
    dthetadIxy = 0.0
    dthetadIyy = 0.0
    tmp = (Ixx - Iyy) ** 2 + (2 * Ixy) ** 2
    # Check for divide by zero
    if tmp != 0.0:
        dthetadIxx = Ixy / tmp * (180 / np.pi)
        dthetadIxy = -(Ixx - Iyy) / tmp * (180 / np.pi)
        dthetadIyy = -Ixy / tmp * (180 / np.pi)

    return dIpmaxdIxx, dIpmaxdIxy, dIpmaxdIyy, dIpmindIxx, dIpmindIxy, dIpmindIyy, dthetadIxx, dthetadIxy, dthetadIyy


def principalAxesSecondMomentArea_b(Ixx, Ixxb, Ixy, Ixyb, Iyy, Iyyb, Ipmaxb, Ipminb, thetab):
    """
    Computes the reverse mode derivatives of principal second moment of area and the angle of the principal axes w.r.t to the second moment of area

    Typically one seed is specified at a time, unless this routine is part of a larger
    chain (remember to zero the seeds as needed).

    Example
    -------
    To compute dIpmax/dIxx (dIpmin/dIxx and dtheta/dIxx are also computed) for example, call
        principalAxesSecondMomentArea_b(Ixx, Ixxb, Ixy, Ixyb, Iyy, Iyyb, Ipmaxb=1.0, Ipminb=0.0, thetab=0.0)

    """

    # Compute the reverse derivatives
    theta2b = 180 * thetab / (2.0 * np.pi)
    iavgb = Ipminb + Ipmaxb
    hypotenuseb = Ipmaxb - Ipminb
    temp = (Ixx - Iyy) / 2.0
    tempb0 = 0.0
    # Check for divide by zero
    tmp = temp**2 + Ixy**2
    if tmp != 0.0:
        tempb0 = hypotenuseb / (2.0 * np.sqrt(tmp))

    tempb = 2.0 * temp * tempb0 / 2.0
    Ixyb = Ixyb + 2.0 * Ixy * tempb0 - 2.0 * (Ixx - Iyy) * theta2b / ((-2.0 * Ixy) ** 2 + (Ixx - Iyy) ** 2)
    Ixxb = Ixxb + tempb
    Iyyb = Iyyb - tempb

    tempb = 2.0 * Ixy * theta2b / ((-2.0 * Ixy) ** 2 + (Ixx - Iyy) ** 2)
    Ixxb = Ixxb + iavgb / 2.0 + tempb
    Iyyb = Iyyb + iavgb / 2.0 - tempb

    # Since we cant update scalar values directly like mutable array we need to return also
    return Ixxb, Ixyb, Iyyb


def sectionModulus(nodes, principalAxes=False):
    """Computes the section modulus of a polygon given a list of (x,y) coordinates in a counterclockwise order
    The section modulus is defined as

    S = Ixx / ymax

    Where the ymax is the maximum distance from the centroid to any node on the polygon.
    If using the minimum principal second moment of area then

     S = Ipmin / ymax

    Parameters
    ----------
    nodes : ndarray (N x 2)
        The (x,y) coordinates of a polygon with N nodes, ordered in a counterclockwise order
    principalAxes : bool
        Use the minimum principal second moment of area instead of Ixx in computing the sectionModulus.
        The minimum principal is used here since for airfoil shaped geometries with axis of origin at centroid

    Returns
    -------
    S : float
        The section modulus of the polygon
    """
    # Before doing anything we shift origin to centroid.
    # This makes principal axis (and other) computation simpler
    xc, yc = centroidPoly(nodes)

    # Make a copy so we are not editing the input
    nodesShifted = np.zeros_like(nodes)
    nodesShifted[:, 0] = nodes[:, 0] - xc
    nodesShifted[:, 1] = nodes[:, 1] - yc

    # Compute the input quantities needed. Since origin is at centroid aboutCentroid has no effect and can be false.
    Ixx, Ixy, Iyy, _ = secondMomentAreaPoly(nodesShifted, aboutCentroid=False)

    if principalAxes:
        _, Ipmin, theta = principalAxesSecondMomentArea(Ixx, Ixy, Iyy)
        Ixx = Ipmin
        # Theta is angle of rotation of max axis, thus we need to add +90 for min axis.
        # The minus is added to rotate the geometry to align with local reference axis
        alpha = -(theta + 90.0)
        M = rotation.rotM(alpha)
        nodesShifted = rotation.rotate(nodesShifted, M)

    # Find the maximum distance in y from centroid. We are at centroid so we can directly use abs.
    yAbs = cs.abs(nodesShifted[:, 1])

    # Compute the ymax using a KS function for a smooth max
    yMax = ksFunction(yAbs, rho=300)

    # Compute the section modulus
    S = Ixx / yMax

    return S


def sectionModulus_b(nodes, nodesb, Sb=1.0, principalAxes=False):
    """
    Computes the reverse mode derivatives of the section modulus w.r.t to the nodes
    """

    # Before doing anything we shift origin to centroid.
    # This makes principal axis (and other) computation simpler
    xc, yc = centroidPoly(nodes)

    # Make a copy so we are not editing the input
    nodesShifted = np.zeros_like(nodes)
    nodesShifted[:, 0] = nodes[:, 0] - xc
    nodesShifted[:, 1] = nodes[:, 1] - yc

    # Compute the input quantities needed. Since origin is at centroid aboutCentroid has no effect and can be false.
    Ixx, Ixy, Iyy, _ = secondMomentAreaPoly(nodesShifted, aboutCentroid=False)
    if principalAxes:
        _, Ipmin, theta = principalAxesSecondMomentArea(Ixx, Ixy, Iyy)
        # Theta is angle of rotation of max axis, thus we need to add +90 for min axis.
        # The minus is added to rotate the geometry to align with local reference axis
        alpha = -(theta + 90.0)
        M = rotation.rotM(alpha)
        nodesShiftedRotated = rotation.rotate(nodesShifted, M)
    else:
        Ipmin = Ixx
        nodesShiftedRotated = nodesShifted

    mask = nodesShiftedRotated[:, 1] >= 0.0
    yAbs = np.where(mask, nodesShiftedRotated[:, 1], -nodesShiftedRotated[:, 1])

    # Compute the ymax using a KS function for a smooth max
    yMax = ksFunction(yAbs, rho=300)

    # Reverse accumulate derivatives
    Ipminb = Sb / yMax
    yMaxb = -(Ipmin * Sb / yMax**2)
    yAbsb = ksFunction_d(yAbs, rho=300) * yMaxb

    invertedMask = np.invert(mask)
    nodesShiftedRotatedb = np.zeros_like(nodesShiftedRotated)
    nodesShiftedRotatedb[:, 1] = np.where(
        invertedMask, nodesShiftedRotatedb[:, 1] - yAbsb, nodesShiftedRotatedb[:, 1] + yAbsb
    )
    yAbsb = 0.0

    nodesShiftedb = np.zeros_like(nodesShifted)
    Ixxb = 0.0
    Ixyb = 0.0
    Iyyb = 0.0
    if principalAxes:
        Mb = np.zeros_like(M)
        rotation.rotate_b(nodesShifted, nodesShiftedb, M, Mb, nodesShiftedRotatedb)
        nodesShiftedRotatedb = 0.0

        alphab = rotation.rotM_b(alpha, Mb)
        thetab = -alphab

        Ixxb, Ixyb, Iyyb = principalAxesSecondMomentArea_b(
            Ixx, Ixxb, Ixy, Ixyb, Iyy, Iyyb, Ipmaxb=0.0, Ipminb=Ipminb, thetab=thetab
        )
    else:
        nodesShiftedb = nodesShiftedRotatedb
        Ixxb = Ipminb

    secondMomentAreaPoly_b(nodesShifted, nodesShiftedb, Ixxb=Ixxb, Ixyb=Ixyb, Iyyb=Iyyb, Jzb=0.0, aboutCentroid=False)

    nodesb[:, 1] = nodesb[:, 1] + nodesShiftedb[:, 1]
    ycb = -np.sum(nodesShiftedb[:, 1])
    nodesShiftedb[:, 1] = 0.0

    nodesb[:, 0] = nodesb[:, 0] + nodesShiftedb[:, 0]
    xcb = -np.sum(nodesShiftedb[:, 0])

    centroidPoly_b(nodes, nodesb, xcb=xcb, ycb=ycb)
    Sb = 0.0


def volumePoly(lowerNodes, upperNodes):
    """
    Compute the volume of a 'polyhedron' defined by a loop of nodes
    on the 'bottom' and a loop on the 'top'.
    Like areaPoly, we use the centroid to split the polygon into triangles.

    .. warning:: This only works for convex polygons
    """

    lc = np.average(lowerNodes, axis=0)
    uc = np.average(upperNodes, axis=0)
    volume = 0.0
    ln = len(lowerNodes)
    n = np.zeros((6, 3))
    for ii in range(len(lowerNodes)):
        # Each "wedge" can be sub-divided to 3 tetrahedra

        # The following indices define the decomposition into three tetrahedra

        # 3 5 4 1
        # 5 2 1 0
        # 0 3 1 5

        #      3 +-----+ 5
        #        |\   /|
        #        | \ / |
        #        |  + 4|
        #        |  |  |
        #      0 +--|--+ 2
        #        \  |  /
        #         \ | /
        #          \|/
        #           + 1
        n[0] = lowerNodes[ii]
        n[1] = lc
        n[2] = lowerNodes[np.mod(ii + 1, ln)]

        n[3] = upperNodes[ii]
        n[4] = uc
        n[5] = upperNodes[np.mod(ii + 1, ln)]
        volume += volumeTetra([n[3], n[5], n[4], n[1]])
        volume += volumeTetra([n[5], n[2], n[1], n[0]])
        volume += volumeTetra([n[0], n[3], n[1], n[5]])

    return volume


def volumeTetra(nodes):
    """
    Compute volume of tetrahedra given by 4 nodes
    """
    a = nodes[1] - nodes[0]
    b = nodes[2] - nodes[0]
    c = nodes[3] - nodes[0]
    # Scalar triple product
    V = (1.0 / 6.0) * np.linalg.norm(np.dot(a, np.cross(b, c)))

    return V


def volumePyramid(a, b, c, d, p):
    """
    Compute volume of a square-based pyramid
    """
    fourth = 1.0 / 4.0

    volume = (
        (p[0] - fourth * (a[0] + b[0] + c[0] + d[0])) * ((a[1] - c[1]) * (b[2] - d[2]) - (a[2] - c[2]) * (b[1] - d[1]))
        + (p[1] - fourth * (a[1] + b[1] + c[1] + d[1]))
        * ((a[2] - c[2]) * (b[0] - d[0]) - (a[0] - c[0]) * (b[2] - d[2]))
        + (p[2] - fourth * (a[2] + b[2] + c[2] + d[2]))
        * ((a[0] - c[0]) * (b[1] - d[1]) - (a[1] - c[1]) * (b[0] - d[0]))
    )

    return volume


def volumePyramid_b(a, b, c, d, p, ab, bb, cb, db, pb):
    """
    Compute the reverse-mode derivative of the square-based
    pyramid. This has been copied from reverse-mode AD'ed tapenade
    fortran code and converted to python to use vectors for the
    points.
    """
    fourth = 1.0 / 4.0
    volpymb = 1.0
    tempb = ((a[1] - c[1]) * (b[2] - d[2]) - (a[2] - c[2]) * (b[1] - d[1])) * volpymb
    tempb0 = -(fourth * tempb)
    tempb1 = (p[0] - fourth * (a[0] + b[0] + c[0] + d[0])) * volpymb
    tempb2 = (b[2] - d[2]) * tempb1
    tempb3 = (a[1] - c[1]) * tempb1
    tempb4 = -((b[1] - d[1]) * tempb1)
    tempb5 = -((a[2] - c[2]) * tempb1)
    tempb6 = ((a[2] - c[2]) * (b[0] - d[0]) - (a[0] - c[0]) * (b[2] - d[2])) * volpymb
    tempb7 = -(fourth * tempb6)
    tempb8 = (p[1] - fourth * (a[1] + b[1] + c[1] + d[1])) * volpymb
    tempb9 = (b[0] - d[0]) * tempb8
    tempb10 = (a[2] - c[2]) * tempb8
    tempb11 = -((b[2] - d[2]) * tempb8)
    tempb12 = -((a[0] - c[0]) * tempb8)
    tempb13 = ((a[0] - c[0]) * (b[1] - d[1]) - (a[1] - c[1]) * (b[0] - d[0])) * volpymb
    tempb14 = -(fourth * tempb13)
    tempb15 = (p[2] - fourth * (a[2] + b[2] + c[2] + d[2])) * volpymb
    tempb16 = (b[1] - d[1]) * tempb15
    tempb17 = (a[0] - c[0]) * tempb15
    tempb18 = -((b[0] - d[0]) * tempb15)
    tempb19 = -((a[1] - c[1]) * tempb15)
    pb[0] = pb[0] + tempb
    ab[0] = ab[0] + tempb16 + tempb11 + tempb0
    bb[0] = bb[0] + tempb19 + tempb10 + tempb0
    cb[0] = cb[0] + tempb0 - tempb11 - tempb16
    db[0] = db[0] + tempb0 - tempb10 - tempb19
    ab[1] = ab[1] + tempb18 + tempb7 + tempb2
    cb[1] = cb[1] + tempb7 - tempb18 - tempb2
    bb[2] = bb[2] + tempb14 + tempb12 + tempb3
    db[2] = db[2] + tempb14 - tempb12 - tempb3
    ab[2] = ab[2] + tempb14 + tempb9 + tempb4
    cb[2] = cb[2] + tempb14 - tempb9 - tempb4
    bb[1] = bb[1] + tempb17 + tempb7 + tempb5
    db[1] = db[1] + tempb7 - tempb17 - tempb5
    pb[1] = pb[1] + tempb6
    pb[2] = pb[2] + tempb13


def volumeHex(x0, x1, x2, x3, x4, x5, x6, x7):
    """
    Evaluate the volume of the hexahedral volume defined by the
    the 8 corners.

    Parameters
    ----------
    x{0:7} : arrays or size (3)
        Array of defining the coordinates of the volume
    """

    p = np.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
    V = 0.0
    V += volumePyramid(x0, x1, x3, x2, p)
    V += volumePyramid(x0, x2, x6, x4, p)
    V += volumePyramid(x0, x4, x5, x1, p)
    V += volumePyramid(x1, x5, x7, x3, p)
    V += volumePyramid(x2, x3, x7, x6, p)
    V += volumePyramid(x4, x6, x7, x5, p)
    V /= 6.0

    return V


def volumeHex_b(x0, x1, x2, x3, x4, x5, x6, x7, x0b, x1b, x2b, x3b, x4b, x5b, x6b, x7b):
    """
    Evaluate the derivative of the volume defined by the 8
    coordinates in the array x.

    Parameters
    ----------
    x{0:7} : arrays of len 3
        Arrays of defining the coordinates of the volume

    Returns
    -------
    xb{0:7} : arrays of len 3
        Derivatives of the volume wrt the points.
    """

    p = np.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
    pb = np.zeros(3)
    volumePyramid_b(x0, x1, x3, x2, p, x0b, x1b, x3b, x2b, pb)
    volumePyramid_b(x0, x2, x6, x4, p, x0b, x2b, x6b, x4b, pb)
    volumePyramid_b(x0, x4, x5, x1, p, x0b, x4b, x5b, x1b, pb)
    volumePyramid_b(x1, x5, x7, x3, p, x1b, x5b, x7b, x3b, pb)
    volumePyramid_b(x2, x3, x7, x6, p, x2b, x3b, x7b, x6b, pb)
    volumePyramid_b(x4, x6, x7, x5, p, x4b, x6b, x7b, x5b, pb)

    pb /= 8.0
    x0b += pb
    x1b += pb
    x2b += pb
    x3b += pb
    x4b += pb
    x5b += pb
    x6b += pb
    x7b += pb


def volumeTriangulatedMesh(p0, p1, p2):
    """
    Compute the volume of a triangulated volume by computing
    the signed areas.

    Parameters
    ----------
    p0, p1, p2 : arrays
        Coordinates of the vertices of the triangulated mesh

    Returns
    -------
    volume : float
        The volume of the triangulated surface

    References
    ----------
    The method is described in, among other places,
    EFFICIENT FEATURE EXTRACTION FOR 2D/3D OBJECTS IN MESH REPRESENTATION
    by Cha Zhang and Tsuhan Chen,
    http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
    """

    volume = (
        np.sum(
            p1[:, 0] * p2[:, 1] * p0[:, 2]
            + p2[:, 0] * p0[:, 1] * p1[:, 2]
            + p0[:, 0] * p1[:, 1] * p2[:, 2]
            - p1[:, 0] * p0[:, 1] * p2[:, 2]
            - p2[:, 0] * p1[:, 1] * p0[:, 2]
            - p0[:, 0] * p2[:, 1] * p1[:, 2]
        )
        / 6.0
    )

    return volume


def volumeTriangulatedMesh_b(p0, p1, p2):
    """
    Compute the gradients of the volume w.r.t. the mesh vertices.

    Parameters
    ----------
    p0, p1, p2 : arrays
        Coordinates of the vertices of the triangulated mesh

    Returns
    -------
    grad_0, grad_1, grad_2 : arrays
        Gradients of volume w.r.t. vertex coordinates
    """

    num_pts = p0.shape[0]
    grad_0 = np.zeros((num_pts, 3))
    grad_1 = np.zeros((num_pts, 3))
    grad_2 = np.zeros((num_pts, 3))

    grad_0[:, 0] = p1[:, 1] * p2[:, 2] - p1[:, 2] * p2[:, 1]
    grad_0[:, 1] = p1[:, 2] * p2[:, 0] - p1[:, 0] * p2[:, 2]
    grad_0[:, 2] = p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]

    grad_1[:, 0] = p0[:, 2] * p2[:, 1] - p0[:, 1] * p2[:, 2]
    grad_1[:, 1] = p0[:, 0] * p2[:, 2] - p0[:, 2] * p2[:, 0]
    grad_1[:, 2] = p0[:, 1] * p2[:, 0] - p0[:, 0] * p2[:, 1]

    grad_2[:, 0] = p0[:, 1] * p1[:, 2] - p0[:, 2] * p1[:, 1]
    grad_2[:, 1] = p0[:, 2] * p1[:, 0] - p0[:, 0] * p1[:, 2]
    grad_2[:, 2] = p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]

    return grad_0 / 6.0, grad_1 / 6.0, grad_2 / 6.0
