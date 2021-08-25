import argparse
import yaml
import json

def get_args():
    parser = argparse.ArgumentParser(description="Applies a set color map to the categories")
    parser.add_argument('coco_path', type=str, help="Path to the coco format dataset")
    parser.add_argument('--colourmap', type=str, default="/opt/habitat_modelling/category_colourmap.yaml", help="Path to yaml format colormap")
    parser.add_argument('--output-file', help="Optional path to the output coco file. If not given, overwrites input")
    args = parser.parse_args()
    return args


def main(args):
    dataset = json.load(open(args.coco_path, 'r'))
    colmap = yaml.load(open(args.colourmap, 'r'))

    if not 'categories' in dataset:
        raise ValueError("There needs to be categories in the dataset")

    for cat in dataset['categories']:
        cid = cat['id']
        colour = colmap[cid]
        cat['color'] = colour

    if args.output_file:
        output_path = args.output_file
    else:
        output_path = args.coco_path

    json.dump(dataset, open(output_path, 'w'), sort_keys=True, indent=4)

if __name__ == "__main__":
    main(get_args())