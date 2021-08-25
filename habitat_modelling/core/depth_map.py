import numpy as np
import math

def ply_to_depth_map(ply_data, resolution):
    x_vals = ply_data['vertex']['x']
    y_vals = ply_data['vertex']['y']
    z_vals = ply_data['vertex']['z']

    num_points = len(x_vals)
    if num_points != len(y_vals) or num_points != len(z_vals):
        raise ValueError("Need to be same amount of x,y,z points")

    min_x = np.min(x_vals)
    min_y = np.min(y_vals)
    max_x = np.max(x_vals)
    max_y = np.max(y_vals)

    true_count = 0
    depth_map = DepthMap(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=resolution, mode="avg")
    for n in range(num_points):
        ret = depth_map.add_point_to_map(x_vals[n], y_vals[n], z_vals[n])
        if ret:
            true_count += 1
    return depth_map


class DepthMap:
    def __init__(self, min_x, min_y, max_x, max_y, resolution, mode='avg'):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.resolution = resolution

        self.grid_width = int(math.ceil((self.max_x - self.min_x)/self.resolution))
        self.grid_height = int(math.ceil((self.max_y - self.min_y)/self.resolution))
        self.grid = np.zeros([self.grid_width, self.grid_height, 2])  # width, height, (mean, variance or num_points)

        if mode not in ['avg', 'var']:
            raise ValueError("Mode has to be one of 'avg', 'var'")
        self.mode = mode


    def add_point_to_map(self, x, y, z, noise=0.0):
        """Adds a point to the map

        Args:
            x (float): The x-coordinate of the point
            y (float): The y-coordinate of the point
            z (float): The z-coordinate of the point
            noise (float): Noise of the point, only used in 'var' mode

        Returns:
            bool: True (succesfully added), False (incompatible - not added)

        """
        if not self._check_within_range(x,y):
            return False
        i,j = self._xy_to_grid(x,y)
        if not self._check_within_grid(i,j):
            return False

        if self.mode == "avg":
            # For mode 0, grid[i,j,1] is num_points
            if self.grid[i,j,1] == 0:
                self.grid[i,j,1] = 1
                self.grid[i,j,0] = z
            else:  # Existing points at this location
                self.grid[i,j,1] += 1
                # Calculate running mean
                old_avg = self.grid[i,j,0]
                self.grid[i,j,0] = old_avg + (z - old_avg)/self.grid[i,j,1]
        elif self.mode == "var":
            raise NotImplementedError("TODO")
        else:
            raise ValueError("Not implemented")
        return True

    def query_grid_at_point(self, x,y,z):
        """Short summary.

        Args:
            x (float): The x-coordinate of the point
            y (float): The y-coordinate of the point
            z (float): The z-coordinate of the point

        Returns:
            list: grid_value, num_points or variance (depending on mode)

        """
        if not self._check_within_range(x,y):
            return
        i,j = self._xy_to_grid(x,y)
        if not self._check_within_grid(i,j):
            return
        return self.grid[i,j,0], self.grid[i,j,1]

    def _grid_to_xy(self, i, j):
        """Converts the grid coordinates back to xy.

        Args:
            i (int): Grid coordinate corresponding to x
            j (int): Grid coordinate corresponding to y

        Returns:
            x,y: The x,y coordinates of that grid point

        """
        x = i*self.resolution + self.min_x
        y = j*self.resolution + self.min_y
        return [x, y]


    def _xy_to_grid(self, x, y):
        """Converts an xy point to grid coordinates

        Args:
            x (float): The x-coordinate of the point
            y (float): The y-coordinate of the point

        Returns:
            int,int: The grid coordinates corresponding to that point

        """
        i = round(int((x - self.min_x)/self.resolution))
        j = round(int((y - self.min_y)/self.resolution))
        return [i, j]

    def _check_within_range(self, x, y):
        """Checks if the points are within the range of the grid

        Args:
            x (float): The x-coordinate of the point
            y (float): The y-coordinate of the point

        Returns:
            bool: True (within range of the grid), False (out of range)

        """
        # Test if point is in bounds
        if x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y:
            return True
        else:
            return False

    def _check_within_grid(self, i, j):
        """Checks if the points are within the range of the grid

        Args:
            i (int): Grid coordinate corresponding to x
            j (int): Grid coordinate corresponding to y

        Returns:
            bool: True (within range of the grid), False (out of range)

        """
        # Test if grid point i,j is in bounds
        if i >= 0 and i < self.grid_width and j >= 0 and j < self.grid_height:
            return True
        else:
            return False


    def sample_depth_map_at_location(self, x, y, sample_shape=None, sample_size=None):
        """
        Samples the depth map centered around x,y.

        Either samples a given shape, e.g. a (400,400) grid, Or a given size, e.g. a (200m),200m grid

        Args:
            x: x-location (centre of sample)
            y: y-location (centre of sample)
            sample_shape: The shape of the returned sample. If this is given, sample_size will be ignored.
            sample_size: The size in meters of the returned sample.

        Returns:
            np.ndarray: The grid sample at the given location
        """
        if not sample_shape and not sample_size:
            raise ValueError("Either a sample_shape or sample_size need to be given")


        # Check the x,y points are in range of the grid
        if not self._check_within_range(x,y):
            return None

        if sample_shape:
            i,j = self._xy_to_grid(x,y)
            min_i = i - int(round(sample_shape[0]/2))
            max_i = min_i + sample_shape[0]
            min_j = i - int(round(sample_shape[1] / 2))
            max_j = min_i + sample_shape[1]

            if self._check_within_grid(min_i, min_j) and self._check_within_grid(max_i, max_j):
                sample = self.grid[min_i:max_i, min_j:max_j]
                return sample
            else:
                return None

        elif sample_size:
            min_x = x - sample_size[0]/2
            min_y = y - sample_size[1]/2
            max_x = x + sample_size[0]/2
            max_y = y + sample_size[1]/2

            min_i, min_j = self._xy_to_grid(min_x, min_y)
            max_i, max_j = self._xy_to_grid(max_x, max_y)

            if self._check_within_grid(min_i, min_j) and self._check_within_grid(max_i, max_j):
                sample = self.grid[min_i:max_i, min_j:max_j]
                return sample
            else:
                return None
        else:
            return None



def remove_depth_map_zeros(dmap):
    """
    Remove all the invalid elements (zero depth) from array and replace with the mean
    Args:
        dmap: (np.ndarray) the depth map grid

    Returns:
        dmap

    """
    dmap = np.nan_to_num(dmap)  # Put NaN values to 0
    # Get the mean of all non-zero values
    vmean = np.mean(dmap[np.nonzero(dmap)])
    # Make all zero values the mean
    dmap[dmap == 0.0] = vmean
    return dmap

def make_depth_map_mean_zero(dmap):
    """
    Subtracts the mean from the array so that the mean is at zezro


    Args:
        dmap:(np.ndarray) the depth map grid

    Returns:
        dmap

    """
    mean = np.mean(dmap)
    dmap = dmap - mean
    return dmap, mean



def plot_depth_map(depth_map, remove_zeros=False, make_zero_mean=False):
    """
    Plots the depth map
    Args:
        depth_map: the DepthMap to be plotted
        remove_zeros: if this is True, make all zeros equal to the mean (makes plotting better).
        make_zero_mean: makes the mean of the array equal to 0

    Returns:

    """
    if remove_zeros:
        dmap = depth_map.grid[:, :, 0]
        dmap = remove_depth_map_zeros(dmap)
        if make_zero_mean:
            dmap, mean = make_depth_map_mean_zero(dmap)
    else:
        dmap = depth_map.grid[:,:,0]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(dmap, cmap='jet', interpolation='nearest', extent=[depth_map.min_x, depth_map.max_x, depth_map.min_y, depth_map.max_y], origin='lower')
    plt.colorbar()
    plt.show()


def process_depth_map_to_array(depth_map):
    """
    Processes the depth map to turn it into an array, with zero mean and no incompatible values.
    Args:
        depth_map:

    Returns:

    """
    dmap = depth_map.grid[:, :, 0]
    dmap = remove_depth_map_zeros(dmap)
    dmap, mean = make_depth_map_mean_zero(dmap)
    return dmap, mean
















