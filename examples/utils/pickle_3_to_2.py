#!/usr/bin/env python3


import argparse
import os
import pickle

def get_args():
    parser = argparse.ArgumentParser(description="Gets a python3 pickle and moves it to protocol 2")
    parser.add_argument('input_pickle', help="Input pickle")
    parser.add_argument('output_pickle', help="Output pickle")
    args = parser.parse_args()
    return args


def main(args):
    data = pickle.load(open(args.input_pickle, 'rb'))
    pickle.dump(data, open(args.output_pickle, 'wb'), protocol=2)

if __name__ == "__main__":
    main(get_args())