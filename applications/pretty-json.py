#!/usr/bin/env python3
import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser(description="Pretty prints a json")
    parser.add_argument('input_json', help="The input json file. If no --output-json is given this file will be overwritten")
    parser.add_argument('--output-json', help="The output json file. If no --output-json is given the input-json will be overwritten")
    args = parser.parse_args()
    return args


def main(args):
    if not os.path.isfile(args.input_json):
        raise OSError("File does not exist")

    js = json.load(open(args.input_json, 'r'))

    if args.output_json:
        output_path = args.output_json
    else:
        output_path = args.input_json

    json.dump(js, open(output_path, 'w'), indent=4, sort_keys=True)


if __name__ == "__main__":
    main(get_args())