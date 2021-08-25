#!/usr/bin/env python3

import argparse
import simplekml
import geojson
from pycocotools.coco import COCO
import webcolors
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm
import pymap3d
import pandas as pd

try:
    from osgeo import gdal
except ImportError:
    import gdal

def get_args():
    parser = argparse.ArgumentParser(description="Creates a geographical output from a coco dataset")
    parser.add_argument('coco_path', help="Path to the coco format dataset")
    parser.add_argument('output', help="Path to the output file, prefixes can be {.kml,.geojson,.csv}")
    args = parser.parse_args()
    return args



class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
      rgba = self.scalarMap.to_rgba(val)
      return (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


def create_kml(args):
    coco = COCO(args.coco_path)

    kml = simplekml.Kml()

    # Create a separate style for each category
    # If color is given in categories
    COL = MplColorHelper('jet',0,len(coco.getCatIds()))

    cat_style_lookup = {}
    for cat in coco.loadCats(coco.getCatIds()):
        if 'color' in cat:
            # hex_color = webcolors.rgb_to_hex((cat['color'][0], cat['color'][1], cat['color'][2]))
            kml_color = simplekml.Color.rgb(cat['color'][0], cat['color'][1], cat['color'][2])
        else:
            rgb = COL.get_rgb(cat['id'])
            kml_color = simplekml.Color.rgb(rgb[0], rgb[1], rgb[2])
            # hex_color = webcolors.rgb_to_hex(tuple(rgb[:3]))

        style = simplekml.Style()
        style.labelstyle.scale = 1
        style.iconstyle.color = kml_color

        style.iconstyle.scale = 1
        cat_style_lookup[cat['id']] = style

    if 'info' in coco.dataset:
        if 'datum_latitude' in coco.dataset['info'] and 'datum_longitude' in coco.dataset['info']:
            datum = {}
            datum['latitude'] = coco.dataset['info']['datum_latitude']
            datum['longitude'] = coco.dataset['info']['datum_longitude']
            if 'datum_altitude' in coco.dataset['info']:
                datum['latitude'] = coco.dataset['info']['datum_altitude']
            else:
                datum['latitude'] = 0.0
        else:
            datum = None

    for image in coco.loadImgs(coco.getImgIds()):
        if 'geo_location' in image:
            geoloc = image['geo_location']
        elif 'pose' in image:
            loc = image['pose']['position']
            geoloc = pymap3d.ned2geodetic(image['pose']['position'][0],
                                          image['pose']['position'][1],
                                          image['pose']['position'][2],
                                          datum['latitude'],
                                          datum['longitude'],
                                          datum['latitude'], ell=None, deg=True)
        # geoloc = [lat, lon, alt]
        ann = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))
        if len(ann) != 0:  # Only plot the point if there is an annotation (i.e. a cluster label).
            pnt = kml.newpoint()
            pnt.coords = [(geoloc[1], geoloc[0], geoloc[2])]
            cat_id = ann[0]['category_id']
            style = cat_style_lookup[cat_id]
            pnt.style = cat_style_lookup[cat_id]


    kml.save(args.output)


def create_geotiff(args):
    raise NotImplementedError("geo-tiff not supported yet")


def create_csv(args):
    coco = COCO(args.coco_path)
    data = {
            'lat': [],
            'lon': [],
            'alt': [],
            'cat': [],
            'cid': [],
            'iid': [],
            'aid': [],
    }

    cid_to_cname = {c['id']: c['name'] for c in coco.loadCats(coco.getCatIds())}

    if 'info' in coco.dataset:
        if 'datum_latitude' in coco.dataset['info'] and 'datum_longitude' in coco.dataset['info']:
            datum = {}
            datum['latitude'] = coco.dataset['info']['datum_latitude']
            datum['longitude'] = coco.dataset['info']['datum_longitude']
            if 'datum_altitude' in coco.dataset['info']:
                datum['latitude'] = coco.dataset['info']['datum_altitude']
            else:
                datum['latitude'] = 0.0
        else:
            datum = None

    for image in coco.loadImgs(coco.getImgIds()):
        if 'geo_location' in image:
            geoloc = image['geo_location']
        elif 'pose' in image:
            loc = image['pose']['position']
            geoloc = pymap3d.ned2geodetic(image['pose']['position'][0],
                                          image['pose']['position'][1],
                                          image['pose']['position'][2],
                                          datum['latitude'],
                                          datum['longitude'],
                                          datum['latitude'], ell=None, deg=True)
        # geoloc = [lat, lon, alt]
        ann = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))
        if len(ann) != 0:  # Only plot the point if there is an annotation (i.e. a cluster label).
            data['lat'].append(geoloc[0])
            data['lon'].append(geoloc[1])
            data['alt'].append(geoloc[2])
            data['cid'].append(ann[0]['category_id'])
            data['cat'].append(cid_to_cname[ann[0]['category_id']])
            data['iid'].append(image['id'])
            data['aid'].append(ann[0]['id'])

    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.output, index=False)

def main(args):

    if args.output.endswith('.kml'):
        create_kml(args)
    elif args.output.endswith('.tiff'):
        create_geotiff(args)
    elif args.output.endswith('.csv'):
        create_csv(args)
    else:
        raise NotImplementedError("Output file format not supported")






if __name__ == "__main__":
    main(get_args())
