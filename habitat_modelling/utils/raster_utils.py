
import pyproj
from affine import Affine
import numpy as np
from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros

try:
    from osgeo import gdal
except ImportError:
    import gdal
from osgeo import osr
import utm
import shapely
import shapely.geometry

def extract_raster_patch(gdal_raster,off_ulx,off_uly,patch_size, band=1):
    columns = patch_size
    rows = patch_size
    patch_data = gdal_raster.GetRasterBand(band).ReadAsArray(off_ulx, off_uly, columns, rows)
    return patch_data


def extract_mosaic_patch(gdal_raster,off_ulx,off_uly,patch_size):
    columns = patch_size
    rows = patch_size

    r = gdal_raster.GetRasterBand(1).ReadAsArray(off_ulx, off_uly, columns, rows)
    g = gdal_raster.GetRasterBand(2).ReadAsArray(off_ulx, off_uly, columns, rows)
    b = gdal_raster.GetRasterBand(3).ReadAsArray(off_ulx, off_uly, columns, rows)
    if r is None or g is None or b is None:
        return None
    patch_data = np.dstack((r,g,b))
    return patch_data


def retrieve_pixel_coords(geo_coord,geot_params):
    x, y = geo_coord[0], geo_coord[1]
    forward_transform =  Affine.from_gdal(*geot_params)
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px = np.around(px).astype(int)
    py = np.around(py).astype(int)
    pixel_coord = px, py
    return pixel_coord


def retrieve_geo_coords(pixel_coord, geot_params):
    i,j = pixel_coord[0], pixel_coord[1]
    forward_transform = Affine.from_gdal(*geot_params)
    x, y = forward_transform * (i, j)
    return x, y


def extract_geotransform(bathy_path, verbose=False):
    # test section of input data
    in_ds = gdal.Open(bathy_path)

    info = {}

    info['driver'] = "{}/{}".format(in_ds.GetDriver().ShortName,
                                    in_ds.GetDriver().LongName)

    info['size'] = [in_ds.RasterXSize, in_ds.RasterYSize,in_ds.RasterCount]

    info['projection'] = in_ds.GetProjection()


    if verbose:
        print("Driver: {}/{}".format(in_ds.GetDriver().ShortName,
                                     in_ds.GetDriver().LongName))

        print("Size is {} x {} x {}".format(in_ds.RasterXSize,
                                            in_ds.RasterYSize,
                                            in_ds.RasterCount))

        print("Projection is {}".format(in_ds.GetProjection()))

    geotransform = in_ds.GetGeoTransform()

    if geotransform:
        info['origin'] = [geotransform[0], geotransform[3]]
        info['pixel_size'] = [geotransform[1], geotransform[5]]
        if verbose:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    #plt.figure(1)
    #plt.imshow(in_ds.GetRasterBand(1).ReadAsArray(), cmap="nipy_spectral")

    return in_ds, geotransform, info


def get_ranges(x_range, y_range, sample_rate, geotransform, dataset_info):
    """
    Selects a subset of pixels to extract when working with a raster patch

    Args:
        x_range: (list) the range of x values to consider (in units matching the raster!)
        y_range: (list) the range of y values to consider (in units matching the raster!)
        sample_rate: (float) the sample rate of pixels. I.e. a sample rate of 0.1 will sample 1 in 10 pixels.
        args: (Argparse.parser) Argparse class containing the input arguments
        geotransform: (list) the geotransform for the raster
        dataset_info: (dict) the dictionary containing the dataset info

    Returns:
        x_samples: (np.ndarray) the samples in the x direction to use (in pixels)
        y_samples: (np.ndarray) the samples in the y direction to use (in pixels)
        num_x_samples: (int) the number of x samples there are
        num_y_samples: (int) the number of y samples there are
        x_range: (list) the [min,max] x values (in units matching the raster!)
        y_range (list) the [min,max] y values (in units matching the raster!)

    """
    # FOR FINDING OUT RANGE https://www.perrygeo.com/python-affine-transforms.html
    if x_range and y_range:
        # Retrieve the corner pixel coordinates
        ul_corner = retrieve_pixel_coords((x_range[0], y_range[1]), geotransform)
        br_corner = retrieve_pixel_coords((x_range[1], y_range[0]), geotransform)

        # Get the max pixel ranges
        x_px_range = [min(ul_corner[0], br_corner[0]), max(ul_corner[0], br_corner[0])]
        y_px_range = [min(ul_corner[1], br_corner[1]), max(ul_corner[1], br_corner[1])]

        # Check the limits
        if x_px_range[0] < 0:
            # If less than limit, set to 0
            x_px_range[0] = 0
            # Also set range to minimum x coordinate
            x_range[0] = float(geotransform[0])
        if x_px_range[1] >= dataset_info['size'][0]:
            # Set to max pixel size
            x_px_range[1] = dataset_info['size'][0]
            # Also set range maximum x to maximum size
            x_range[1] = float(geotransform[0]) + float(geotransform[1])*float(dataset_info['size'][0])  # Assumes no rotation
        if y_px_range[0] < 0:
            # If less than limit, set to 0
            y_px_range[0] = 0
            # Also set range minimum y to minimum
            # y_range[0] = float(geotransform[3])  # problem with this
            y_range[0] = float(geotransform[3]) - float(geotransform[5]) * float(dataset_info['size'][1])

        if y_px_range[1] >= dataset_info['size'][1]:
            y_px_range[1] = dataset_info['size'][1]
            # Also set range maximum y to maximum size
            y_range[1] = float(geotransform[3])

        num_x_samples = int((x_px_range[1] - x_px_range[0])*sample_rate)
        num_y_samples = int((y_px_range[1] - y_px_range[0])*sample_rate)

        x_samples = np.linspace(x_px_range[0], x_px_range[1], num_x_samples)
        y_samples = np.linspace(y_px_range[0], y_px_range[1], num_y_samples)


    elif (x_range and not y_range) or (not x_range and y_range):
        raise ValueError("X Range and Y Range have to be given")
    else:
        num_x_samples = int(dataset_info['size'][0]*sample_rate)
        num_y_samples = int(dataset_info['size'][1]*sample_rate)
        x_samples = np.linspace(0, dataset_info['size'][0], num_x_samples)
        y_samples = np.linspace(0, dataset_info['size'][1], num_y_samples)

        x_range = (float(geotransform[0]), float(geotransform[0]) + float(geotransform[1]) * float(num_x_samples))
        # y_range = (float(geotransform[3]), float(geotransform[3]) + float(geotransform[5]) * float(num_y_samples)) # TODO problem with this
        y_range = (float(geotransform[3]) - float(geotransform[5]) * float(num_y_samples), float(geotransform[3]))

    # x_samples, y_samples should be in pixels!
    return x_samples, y_samples, num_x_samples, num_y_samples, x_range, y_range



def get_lla_from_pixels(pixel_coords, geotransform, projection):
    """
    Transforms the pixel coordinates from the raster to geo coordinates (lat lon)
    Args:
        pixel_coords: the pixel coordinates
        geotransform: the geotransform
        projection: the projection - get it from gdal dataset info

    Returns:
        lla: (lat, lon, alt)
    """
    # Get the world coordinates of x,y samples

    gx, gy = retrieve_geo_coords(pixel_coords, list(geotransform))
    if 'UTM' in projection:
        srs = osr.SpatialReference(wkt=projection)
        projcs = srs.GetAttrValue('projcs')
        if 'zone' in projcs:
            zonestr = projcs.split(' ')[-1]
            zone_num = int(zonestr[:2])
            zone_hem = zonestr[-1]
            if zone_hem == "N":
                northern = True
            elif zone_hem == "S":
                northern = False
            else:
                raise ValueError("Zone hemisphere has to be either north or south")
        else:
            raise ValueError("Projection doesn't contain zone")
        latlon = list(utm.to_latlon(gx, gy, zone_num, None, northern=northern))
    else:
        latlon = [gy, gx]  # TODO check this output
    lla = latlon + [0.]
    return lla


def get_pixels_from_lla(lla, geotransform, projection):
    if 'UTM' in projection:  # TODO find less hacky way to determine
        use_utm = True
    else:
        use_utm = False

    if use_utm:
        ux, uy, zone_num, zone_letter = utm.from_latlon(lla[0], lla[1])
        px, py = retrieve_pixel_coords([ux, uy], list(geotransform))
    else:
        px, py = retrieve_pixel_coords([lla[1], lla[0]], list(geotransform))

    return px, py


def get_bathy_patch(geo_ds, x, y, size, no_data_val):
    """
    Gets the dmap and depth from the bathymetry raster
    Args:
        geo_ds: (gdal Dataset) The raster
        x: (int) The x pixel coordinate
        y: (int) The y pixel coordinate
        size: (tuple) The size of the bathymetry patch to extract
        no_data_val: (depends) The no data value of the raster. Type depends on type of the raster.

    Returns:
        dmap: The depth map with 0 mean
        depth: The mean depth
    """
    patch = extract_raster_patch(geo_ds, int(x), int(y), int(size[0]))
    if patch is not None:
        if np.any(patch == no_data_val) or np.any(np.isnan(patch)) or np.any(np.isinf(patch)) or np.max(patch) > 1e8 or np.min(patch) < -1e8:
            patch = None

    if patch is not None:  # Check the patch is ok
        dmap, depth = make_depth_map_mean_zero(remove_depth_map_zeros(patch))
    else:
        dmap = None
        depth = None

    return dmap, depth

def get_raster_patch(geo_ds, x, y, size, no_data_val, band=1):
    """
    Gets the dmap and depth from the bathymetry raster
    Args:
        geo_ds: (gdal Dataset) The raster
        x: (int) The x pixel coordinate
        y: (int) The y pixel coordinate
        size: (tuple) The size of the bathymetry patch to extract
        no_data_val: (depends) The no data value of the raster. Type depends on type of the raster.

    Returns:
        patch: The raster patch.
    """
    patch = extract_raster_patch(geo_ds, int(x), int(y), int(size[0]), band=band)
    if patch is not None:
        if np.any(patch == no_data_val) or np.any(np.isnan(patch)) or np.any(np.isinf(patch)):
            patch = None
    return patch

def get_multiband_raster_patch(geo_ds, x, y, size, no_data_val, bands):
    patches = []
    for band in bands:
        patch = get_raster_patch(geo_ds, x, y, size, no_data_val, band)
        if patch is None:
            return None
        patches.append(patch)
    return np.concatenate(patches, axis=-1)



def raster_collection_from_cfg(cfg, lib='torch'):
    """
    Generates a raster collection dictionary from the configuration file.

    Args:
        cfg: the 'rasters' dictionary section of the cfg

    Returns:
        dict: The raster collection dictionary
    """
    if lib == 'torch':
        from habitat_modelling.ml.torch.transforms.bathy_transforms import to_float_tensor, to_tensor
    else:
        raise NotImplementedError("Only torch lib supported")

    raster_collection = {}
    for key, value in cfg.items():
        # Check the raster is in the dataset

        raster_ds, geotransform, raster_info = extract_geotransform(value['path'])

        raster_collection[key] = {
            "dataset": raster_ds,
            "info": raster_info,
            "geotransform": geotransform,
            "size": cfg['size']
        }
        if 'boundary' in value:
            if len(value['boundary']) == 4:
                boundary = shapely.geometry.box(value['boundary'])
            else:
                boundary = shapely.geometry.Polygon(value['boundary'])
            raster_collection[key]['boundary'] = boundary
        else:
            raster_collection[key]['boundary'] = None
        if value.get('transform', 'tensor') == 'tensor':
            raster_collection[key]['transform'] = to_tensor
        elif value['transform'] == 'float_tensor':
            raster_collection[key]['transform'] = to_float_tensor
        else:
            raise NotImplementedError("Only tensor and float_tensor supported")
    return raster_collection

def raster_lists_from_cfg(cfg, lib='torch'):
    """
    Generate lists to be used with rasters.
    Args:
        cfg: the 'rasters' section of the cfg

    Returns:
        [list, list, list, list]: raster_names, raster_paths, raster_sizes, raster_boundaries, raster_transforms
    """
    if lib == 'torch':
        from habitat_modelling.ml.torch.transforms.bathy_transforms import to_float_tensor, to_tensor
    else:
        raise NotImplementedError("Only torch lib supported")
    name_list = []
    path_list = []
    size_list = []
    transform_dict = {}
    boundary_list = []
    for key, value in cfg.items():
        # Check the raster is in the dataset
        name_list.append(key)
        path_list.append(value['path'])
        size_list.append(value['size'])
        if 'boundary' in value:
            if value['boundary'] is None:
                boundary = None
            elif len(value['boundary']) == 4:
                boundary = shapely.geometry.box(value['boundary'])
            else:
                boundary = shapely.geometry.Polygon(value['boundary'])
            boundary_list.append(boundary)
        else:
            boundary_list.append(None)

        if value.get('transform', 'tensor') == 'tensor':
            transform_dict[key] = to_tensor
        elif value['transform'] == 'float_tensor':
            transform_dict[key] = to_float_tensor
        else:
            raise NotImplementedError("Only tensor and float_tensor supported")
    return name_list, path_list, size_list, boundary_list, transform_dict


def load_raster_registry(raster_cfg):
    raster_registry = {}
    for key,entry in raster_cfg.items():
        raster_ds, geotransform, raster_info = extract_geotransform(entry['path'])
        nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()
        raster_registry[key] = {
            "dataset": raster_ds,
            "info": raster_info,
            "geotransform": geotransform,
            "size": entry['size'],
            "no_data_val": nodataval,
            "path": entry['path'],
            "band_count": raster_ds.RasterCount
        }
    return raster_registry