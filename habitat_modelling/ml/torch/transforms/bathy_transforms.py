import torch
import numpy as np

def create_bathy_transform_list_from_args(args):
    """
    Creates the bathymetry transform lists
    Args:
        args: (ArgumentParser) the argument parser class

    Returns:
        [preprocessing_transforms, post_processing_transforms]

    """
    # Create the bathymetry transforms
    bathy_transforms = []
    if args.bathy_transforms:
        for op in args.bathy_transforms.split(','):
            if op.lower() == "model":
                raise NotImplementedError("TODO")
            if op.lower() == "float" or op.lower() == "tensor":
                bathy_transforms.append(transforms.ToTensor())
    if len(bathy_transforms) == 0:
        bathy_transforms = None

    return [transforms.Compose(bathy_transforms), _]


def to_float_tensor(x):
    if x.dtype == np.uint8:  # Normalise if an integer
        x = x / 255
    return torch.FloatTensor(x)

def to_tensor(x):
    return torch.tensor(x)
