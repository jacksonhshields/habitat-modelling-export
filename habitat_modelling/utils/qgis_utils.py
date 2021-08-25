import numpy as np
import os
import warnings
import geopandas as gpd
import shapely
import shapely.geometry
import pandas as pd

def get_colour_ramps(ramp, classes=-1):
    viridis = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
    magma = ['#000003', '#3B0F6F', '#8C2980', '#DD4968', '#FD9F6C', '#FBFCBF']
    category10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    category20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    set1 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    # from bokeh.palettes
    if ramp == "viridis":
        return viridis
    elif ramp == "magma":
        return magma
    elif ramp == "category10":
        if classes != -1:
            return category10[:classes]
        else:
            return category10
    elif ramp == "category20":
        if classes != -1:
            return category20[:classes]
        else:
            return category20
    elif ramp == "set1":
        if classes != -1:
            return set1[:classes]
        else:
            return set1
    elif ramp == "turbo":
        import bokeh.palettes
        if classes != -1:
            return bokeh.palettes.Turbo[classes]
        else:
            return bokeh.palettes.Turbo[256]
    else:
        raise ValueError("Colour ramp option %s not available" %(ramp))


def write_geodf_to_file(geodf, output_file, output_line_file=None, input_crs="EPSG:4326"):
    geodf.to_file(output_file)

    if output_line_file is not None:
        ls = shapely.geometry.LineString(geodf['geometry'])
        line_series = gpd.GeoSeries(ls, crs=input_crs)
        line_series.to_file(output_line_file)


def csv_to_shp(csv_file, output_file, output_line_file=None, input_crs="EPSG:4326"):
    """
    Turns a CSV into a shapefile, with an optional path as well.
    """
    df = pd.read_csv(csv_file)
    geometry = gpd.points_from_xy(df.lon, df.lat)
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs=input_crs)
    write_geodf_to_file(geo_df, output_file, output_line_file, input_crs)


def lls_to_shp(lls, output_file, output_line_file=None):
    """
    Turns a list of lls into a shapefile and an optional path shapefile as well.
    """
    lats = []
    lons = []
    for ll in lls:
        lats.append(ll[0])
        lons.append(ll[1])
    df = pd.DataFrame.from_dict({'lat': lats, 'lon': lons})
    input_crs = "EPSG:4326"
    geometry = gpd.points_from_xy(lons, lats)
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs=input_crs)
    write_geodf_to_file(geo_df, output_file, output_line_file, input_crs)


def multiple_paths_to_shp(multi_lls, output_file, output_line_file):
    """
    Turns multiple paths into a single shapefile.

    Args:
        multi_lls: [[np.ndarray]] or [np.ndarray] A list of multiple paths
        output_file: (str) the output shapefile
        output_line_file: (str) The output shapefile for outputting lines.
    Returns:
        None
    """
    all_lats = []
    all_lons = []

    if len(multi_lls) == 0:
        warnings.warn("No paths provided")
        return
    all_lines = []
    for n, path in enumerate(multi_lls):  # For each path in the list

        # Accumulate the lat lons
        if type(path) is list:
            lats = []
            lons = []
            for ll in path:
                lats.append(ll[0])
                lons.append(ll[1])
        else:  # should be np.ndarray
            lats = path[:,0].tolist()
            lons = path[:,1].tolist()
        df = pd.DataFrame.from_dict({'lat': lats, 'lon': lons})
        input_crs = "EPSG:4326"
        geometry = gpd.points_from_xy(lons, lats)
        geodf = gpd.GeoDataFrame(df, geometry=geometry, crs=input_crs)
        ls = shapely.geometry.LineString(geodf['geometry'])
        all_lines.append(ls)
        # line_series = gpd.GeoSeries(ls, crs=input_crs)
        # if n == 0:
        #     combined_series = line_series
        # else:
        #     combined_series.append(line_series)
        #
        all_lats.extend(lats)
        all_lons.extend(lons)
    multiline = shapely.geometry.MultiLineString(all_lines)
    combined_series = gpd.GeoSeries(multiline, crs=input_crs)
    combined_series.to_file(output_line_file)
    all_df = pd.DataFrame.from_dict({'lat': all_lats, 'lon': all_lons})
    input_crs = "EPSG:4326"
    geometry = gpd.points_from_xy(all_lons, all_lats)
    all_gdf = gpd.GeoDataFrame(all_df, geometry=geometry, crs=input_crs)
    all_gdf.to_file(output_file)


def get_overall_extent(all_extents):
    """
    Gets the overall extent for all the layers

    Args:
        all_extents: [QgsRectangle] A list of all the rectangles

    Returns:
        QgsRectangle: The overall extent
    """
    overall_extent = None
    for n, extent in enumerate(all_extents):
        if n == 0:
            overall_extent = extent
        else:
            if extent.xMaximum() > overall_extent.xMaximum() and not np.isnan(extent.xMaximum()):
                overall_extent.setXMaximum(extent.xMaximum())
            if extent.xMinimum() < overall_extent.xMinimum() and not np.isnan(extent.xMinimum()):
                overall_extent.setXMinimum(extent.xMinimum())
            if extent.yMaximum() > overall_extent.yMaximum() and not np.isnan(extent.yMaximum()):
                overall_extent.setYMaximum(extent.yMaximum())
            if extent.yMinimum() < overall_extent.yMinimum() and not np.isnan(extent.yMinimum()):
                overall_extent.setYMinimum(extent.yMinimum())

    return overall_extent



def create_qgis_project(save_path, raster_config=None, vector_config=None, layout_config=None):
    """
    Creates a QGIS project from the given rasters/vectors.

    Args:
        save_path: (str) The path to the output qgis project
        raster_config: (dict) key: raster_name, entry: (dict) configuration of the raster
            entry: {
                type: Options {bathymetry, categorical, uncertainty, entropy
                path: The path to the raster [required]
                colour_map: The colour map to use, options such as viridis, turbo, category10... [optional]
                num_classes: For a categorical raster, defines the number of classes to use. If left blank uses the max value. [optional]

            }
        vector_config:

    Returns:

    """
    from qgis.core import (
        QgsProject,
        QgsPrintLayout,
        QgsLayoutPoint,
        QgsLayoutSize,
        QgsUnitTypes,
        QgsLayoutItemLegend,
        QgsLayerTree,
        QgsLayoutItemLabel,
        QgsMapSettings,
        QgsCoordinateTransform


    )

    from qgis.gui import (
        QgsLayerTreeMapCanvasBridge,
        QgsMapCanvas,

    )
    import os
    from qgis.core import (
        QgsVectorLayer,
        QgsRasterLayer,
        QgsApplication,
        QgsProviderRegistry,
        QgsRasterBandStats,
        QgsColorRampShader,
        QgsRasterShader,
        QgsSingleBandPseudoColorRenderer,
        QgsCoordinateReferenceSystem,
        QgsLayoutItemMap,
        QgsMapRendererParallelJob,
    )
    from PyQt5.QtGui import QColor, QFont
    from PyQt5.QtCore import QSize

    from qgis.utils import iface

    # Initialise QGIS
    QgsApplication.setPrefixPath("/usr", True)
    qgs = QgsApplication([], False)
    qgs.initQgis()
    provider_registry = QgsProviderRegistry.instance()
    if not 'ogr' in provider_registry.providerList():
        print('Could not find OGR provider!')
    else:
        print('Providers found ok!')
    # Create QGIS project
    project = QgsProject.instance()
    canvas = QgsMapCanvas()
    # canvas.show()
    bridge = QgsLayerTreeMapCanvasBridge(project.layerTreeRoot(), canvas)
    overall_crs = QgsCoordinateReferenceSystem(4326)
    project.setCrs(overall_crs)
    project.write(save_path)

    all_extents = []

    layers = {}

    # Add rasters
    if raster_config is not None:
        for raster_name, raster_entry in raster_config.items():
            layers[raster_name] = QgsRasterLayer(raster_entry['path'], raster_name)
            provider = layers[raster_name].dataProvider()
            extent = layers[raster_name].extent()
            stats = provider.bandStatistics(1, QgsRasterBandStats.All, extent, 0)
            # canvas.setExtent(extent)

            # If the CRS doesn't match the projects, transform it before adding to extents
            if layers[raster_name].crs() == overall_crs:
                all_extents.append(extent)
            else:
                transformContext = QgsProject.instance().transformContext()
                xform = QgsCoordinateTransform(layers[raster_name].crs(), overall_crs, transformContext)
                trans_extent = xform.transform(extent)
                print("TransformedExtent", trans_extent.toString())
                all_extents.append(trans_extent)

            if raster_entry['type'] == "bathymetry" or raster_entry['type'] == "bathymetry":
                colour_ramp = get_colour_ramps(raster_entry.get("colour_ramp", "viridis"))
                if raster_entry.get('negative_depth', False):
                    depth_range = np.linspace(stats.minimumValue, 0.0, len(colour_ramp))
                else:
                    depth_range = np.linspace(0.0, stats.maximumValue, len(colour_ramp))[::-1]
                ramp_list = []
                for hexcol, depth in zip(colour_ramp, depth_range):
                    ramp_list.append(QgsColorRampShader.ColorRampItem(depth, QColor(hexcol)))

                fcn = QgsColorRampShader()
                fcn.setColorRampType(QgsColorRampShader.Interpolated)
                fcn.setClassificationMode(QgsColorRampShader.Continuous)
                fcn.setColorRampItemList(ramp_list)
                shader = QgsRasterShader()
                shader.setRasterShaderFunction(fcn)
                renderer = QgsSingleBandPseudoColorRenderer(layers[raster_name].dataProvider(), 1, shader)
                layers[raster_name].setRenderer(renderer)
                layers[raster_name].triggerRepaint()


            elif raster_entry['type'] == "uncertainty" or raster_entry['type'] == "entropy":
                colour_ramp = get_colour_ramps(raster_entry.get("colour_ramp", "viridis"))
                val_range = np.linspace(0.0, stats.maximumValue, len(colour_ramp))[::-1]
                ramp_list = []
                for hexcol, depth in zip(colour_ramp, val_range):
                    ramp_list.append(QgsColorRampShader.ColorRampItem(depth, QColor(hexcol)))

                fcn = QgsColorRampShader()
                fcn.setColorRampType(QgsColorRampShader.Interpolated)
                fcn.setColorRampItemList(ramp_list)
                fcn.setClassificationMode(QgsColorRampShader.Continuous)
                shader = QgsRasterShader()
                shader.setRasterShaderFunction(fcn)
                renderer = QgsSingleBandPseudoColorRenderer(layers[raster_name].dataProvider(), 1, shader)
                layers[raster_name].setRenderer(renderer)
                layers[raster_name].triggerRepaint()
            elif raster_entry['type'] == "categorical":
                num_classes = int(raster_entry.get("num_classes", stats.maximumValue+1))
                print("Num classes", num_classes)
                if num_classes <= 20:
                    ramp_type = raster_entry.get("colour_ramp", "category")
                else:
                    ramp_type = raster_entry.get("colour_ramp", "turbo")
                interpolate = False
                if ramp_type == "category":
                    if num_classes <= 10:
                        colour_ramp = get_colour_ramps("category10", classes=num_classes)
                    elif num_classes < 20:
                        colour_ramp = get_colour_ramps("category20", classes=num_classes)
                    else:
                        raise ValueError("Maximum classes for category datasets is 20, use turbo instead")
                elif ramp_type == "set":
                    if num_classes <= 9:
                        colour_ramp = get_colour_ramps("set1", classes=num_classes)
                    else:
                        raise ValueError("Too many classes for set1")
                elif ramp_type == "turbo":
                    if num_classes <= 11 and num_classes >= 3:
                        colour_ramp = get_colour_ramps("turbo", classes=num_classes)
                    else:
                        colour_ramp = get_colour_ramps("turbo")
                        interpolate = True
                else:
                    raise ValueError("Colour map format %s not available" % ramp_type)
                if interpolate:
                    val_range = np.linspace(0.0, float(num_classes), len(colour_ramp))[::-1]
                    ramp_list = []
                    for hexcol, val in zip(colour_ramp, val_range):
                        ramp_list.append(QgsColorRampShader.ColorRampItem(val, QColor(hexcol)))
                    fcn = QgsColorRampShader()
                    fcn.setColorRampType(QgsColorRampShader.Interpolated)
                    fcn.setColorRampItemList(ramp_list)
                    fcn.setClassificationMode(QgsColorRampShader.Continuous)

                else:
                    ramp_list = []
                    for hexcol, cl in zip(colour_ramp, range(num_classes)):
                        ramp_list.append(QgsColorRampShader.ColorRampItem(cl, QColor(hexcol)))
                    fcn = QgsColorRampShader()
                    fcn.setColorRampType(QgsColorRampShader.Interpolated)
                    fcn.setColorRampItemList(ramp_list)
                    fcn.setClassificationMode(QgsColorRampShader.Continuous)

                shader = QgsRasterShader()
                shader.setRasterShaderFunction(fcn)
                renderer = QgsSingleBandPseudoColorRenderer(layers[raster_name].dataProvider(), 1, shader)
                layers[raster_name].setRenderer(renderer)
                layers[raster_name].triggerRepaint()
            else:
                raise ValueError("Raster type is invalid: raster: %s, type: %s" %(raster_name, raster_entry['type']))

            project.addMapLayer(layers[raster_name])

    # Add vectors
    if vector_config is not None:
        for vector_name, vector_entry in vector_config.items():
            if vector_entry['path'].endswith('.csv'):
                # uri = "file:///" + vector_entry['path'] + "?delimiter=%s&crs=epsg:4326&xField=%s&yField=%s&spatialIndex=no&subsetIndex=no&watchFile=no" % (",", vector_entry['longitude_field'], vector_entry['latitude_field'])
                uri = "file:///" + vector_entry['path'] + "?type=csv&detectTypes=yes&xField=lon&yField=lat&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no"
                layers[vector_name] = QgsVectorLayer(uri, vector_name, 'delimitedtext')
                layers[vector_name].setCrs( QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId) )
                if not layers[vector_name].isValid():
                    warnings.warn("Failed to add {} with URI = {}".format(vector_name, uri))
                else:
                    project.addMapLayer(layers[vector_name])
            elif vector_entry['path'].endswith('.shp'):
                uri = "file:///" + vector_entry['path']
                layers[vector_name] = QgsVectorLayer(vector_entry['path'], vector_name, "ogr")
                if not layers[vector_name].isValid():
                    warnings.warn("Failed to add {} with path = {}".format(vector_name, vector_entry['path']))
                else:
                    project.addMapLayer(layers[vector_name])
            else:
                raise NotImplementedError("Only CSV loading currently implemented")

            if layers[vector_name].isValid():
                extent = layers[vector_name].extent()
                all_extents.append(extent)
    overall_extent = get_overall_extent(all_extents)
    print(overall_extent.toString())
    # canvas.show()
    canvas.refreshAllLayers()

    canvas.setExtent(overall_extent)

    if layout_config is not None:
        # https://data.library.virginia.edu/how-to-create-and-export-print-layouts-in-python-for-qgis-3/
        manager = project.layoutManager()

        layout = QgsPrintLayout(project)
        layoutName = "PrintLayout"

        # initializes default settings for blank print layout canvas
        layout.initializeDefaults()

        layout.setName(layoutName)
        layouts_list = manager.printLayouts()
        for layout in layouts_list:
            if layout.name() == layoutName:
                manager.removeLayout(layout)
        manager.addLayout(layout)

        map = QgsLayoutItemMap(layout)
        map.setCrs(overall_crs)

        # I have no idea what this does, but it is necessary
        # map.setRect(20, 20, 20, 20)

        # Move & Resize map on print layout canvas
        map.attemptMove(QgsLayoutPoint(5, 27, QgsUnitTypes.LayoutMillimeters))
        map.attemptResize(QgsLayoutSize(239, 178, QgsUnitTypes.LayoutMillimeters))

        # Set Map Extent
        # defines map extent using map coordinates
        map.setExtent(overall_extent)
        layout.addLayoutItem(map)

        legend = QgsLayoutItemLegend(layout)
        legend.setTitle("Legend")
        layout.addLayoutItem(legend)
        legend.attemptMove(QgsLayoutPoint(246, 5, QgsUnitTypes.LayoutMillimeters))

        # TODO fix this to remove checked layers
        # Checks layer tree objects and stores them in a list. This includes csv tables
        # checked_layers = [layer.name() for layer in QgsProject().instance().layerTreeRoot().children() if
        #                   layer.isVisible()]
        # print(f"Adding {checked_layers} to legend.")
        # # get map layer objects of checked layers by matching their names and store those in a list
        # layersToAdd = [layer for layer in QgsProject().instance().mapLayers().values() if
        #                layer.name() in checked_layers]
        # legend = QgsLayoutItemLegend(layout)
        # legend.setTitle("Legend")
        # root = QgsLayerTree()
        # for layer in layersToAdd:
        #     # add layer objects to the layer tree
        #     root.addLayer(layer)
        # legend.model().setRootGroup(root)
        # layout.addLayoutItem(legend)
        # legend.attemptMove(QgsLayoutPoint(246, 5, QgsUnitTypes.LayoutMillimeters))

        title_text = layout_config.get("title", None)
        if title_text is not None:
            title = QgsLayoutItemLabel(layout)
            title.setText(layout_config.get("title", None))
            title.setFont(QFont(layout_config.get("title_font", "Arial"), layout_config.get("title_fontsize", 28)))
            title.adjustSizeToText()
            layout.addLayoutItem(title)
            title.attemptMove(QgsLayoutPoint(10, 4, QgsUnitTypes.LayoutMillimeters))

        subtitle_text = layout_config.get("subtitle", None)
        if subtitle_text is not None:
            subtitle = QgsLayoutItemLabel(layout)
            subtitle.setText(layout_config.get("title", None))
            subtitle.setFont(QFont(layout_config.get("title_font", "Arial"), layout_config.get("title_fontsize", 16)))
            subtitle.adjustSizeToText()
            layout.addLayoutItem(subtitle)
            subtitle.attemptMove(QgsLayoutPoint(11, 20, QgsUnitTypes.LayoutMillimeters))  # allows moving text box


        # TODO add grids



    canvas.refreshAllLayers()
    project.write()
    qgs.exitQgis()


def create_qgis_project_from_inference_folder(folder, save_path="inference_generated.qgs", bathymetry=None, backscatter=None):
    raster_config = {}
    for raster in ["var", "entropy", "uncertainty", "aleatoric", "epistemic"]:
        path = os.path.join(folder, raster + "_map.tif")
        if not os.path.isfile(path):
            continue
        entry = {
            "path": path,
            "colour_map": "viridis",
            "type": "uncertainty"
        }
        raster_config[raster] = entry
    path = os.path.join(folder, "category_map.tif")
    if os.path.isfile(path):
        entry = {
            "path": path,
            "colour_map": "category",
            "type": "categorical"
        }
        raster_config["category"] = entry

    if bathymetry:
        entry = {
            "path": bathymetry,
            "colour_map": "viridis",
            "type": "bathymetry",
        }
        raster_config["bathymetry"] = entry
    if os.path.isabs(save_path):
        qgis_project_file = save_path
    else:
        qgis_project_file = os.path.join(folder, save_path)
    create_qgis_project(qgis_project_file, raster_config=raster_config)












