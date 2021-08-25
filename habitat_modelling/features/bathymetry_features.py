import numpy as np


def triangle3d_area(a, b, c):
    """
    Calculates the area of triangle with points A, B, C.

    https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
    """
    l_ab = np.sqrt(np.sum((a - b) ** 2))
    l_ac = np.sqrt(np.sum((a - c) ** 2))
    theta = np.arccos(np.dot(a - b, a - c) / (l_ab * l_ac))
    area = 0.5 * l_ab * l_ac * np.sin(theta)
    return area


def surface_area(dmap, geotransform):
    """
    Calculate the surface area of a dmap by making triangles over the surface.

    For every point. Makes two triangles and finds the area of each.

    Args:
        dmap: (np.ndarray) The patch to calculate the statistics on
        geotransform: (list) The geotransform list from GDAL
    Returns:
        (float): The surface area

    """
    x_points = np.linspace(0, geotransform[1] * dmap.shape[0], dmap.shape[0])
    y_points = np.linspace(0, geotransform[5] * dmap.shape[1], dmap.shape[1])

    surface_area = 0

    for i in range(1, x_points.shape[0]):
        for j in range(1, y_points.shape[0]):
            # Triangle 1
            pa = np.array([x_points[i - 1], y_points[j - 1], dmap[i - 1, j - 1]])
            pb = np.array([x_points[i], y_points[j - 1], dmap[i - 1, j]])
            pc = np.array([x_points[i], y_points[j], dmap[i, j]])
            area1 = triangle3d_area(pa, pb, pc)
            # Triangle 2
            pa = np.array([x_points[i - 1], y_points[j - 1], dmap[i - 1, j - 1]])
            pb = np.array([x_points[i - 1], y_points[j], dmap[i, j - 1]])
            pc = np.array([x_points[i], y_points[j], dmap[i, j]])
            area2 = triangle3d_area(pa, pb, pc)

            area = area1 + area2
            surface_area += area
    return surface_area


def calc_plane_theta(dmap, geotransform):
    """
    Fits a plane to the depth map.

    https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array

    Args:
        dmap: (np.ndarray) The patch to calculate the statistics on
        geotransform: (list) The geotransform list from GDAL

    Returns:
        plane: (np.ndarray) The plane that is fit to the depth map
        theta: (np.ndarray) The normal vector used.
    """
    x_points = np.linspace(0, geotransform[1] * dmap.shape[0], dmap.shape[0]) - (dmap.shape[0] * geotransform[1]) / 2
    y_points = np.linspace(0, geotransform[5] * dmap.shape[1], dmap.shape[1]) - (dmap.shape[1] * geotransform[5]) / 2

    m = dmap.shape[0]

    X1, X2 = np.meshgrid(x_points, y_points)
    Y = dmap
    # Regression
    X = np.hstack((np.reshape(X1, (m * m, 1)), np.reshape(X2, (m * m, 1))))
    X = np.hstack((np.ones((m * m, 1)), X))
    YY = np.reshape(Y, (m * m, 1))

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    plane = np.reshape(np.dot(X, theta), (m, m))

    return plane, theta


def calc_rugosity(dmap, geotransform):
    """
    Calculates the rugosity using the formula:

    rugosity = Area(depth_map)/Area(plane)

    Args:
        dmap:
        geotransform:

    Returns:
        rugosity: (float) The rugosity

    """
    dmap_surface_area = surface_area(dmap, geotransform)
    plane, theta = calc_plane_theta(dmap, geotransform)
    plane_surface_area = surface_area(plane, geotransform)  # TODO make more efficient. Use the Theta
    rugosity = dmap_surface_area/plane_surface_area
    return rugosity