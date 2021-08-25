import numpy as np
import keras.models

class FourierFeatures:
    """
    Resizes an image
    """
    def __init__(self, n_ffts):
        """
        Resizes an image
        Args:
            n_ffts: [n_ffts_x,n_ffts_y] ints
        """
        if type(n_ffts) == int:
            self.n_ffts = [n_ffts, n_ffts]
        else:
            self.n_ffts = list(n_ffts)

    def transform(self, dmap):
        f = np.fft.fft2(dmap, self.n_ffts)
        fshift = np.fft.fftshift(f)
        return fshift.real


class ModelFeatures:
    """
    Gets features from a model
    """
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def transform(self, dmap):
        pred = self.model.predict_on_batch(np.expand_dims(np.expand_dims(dmap, axis=2),axis=0))
        return np.squeeze(pred,axis=0)


class NoOp:
    def __init__(self):
        return
    def transform(self, dmap):
        return dmap


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
            if op.lower() == "fourier":
                n_ffts = [10,10]
                bathy_shape = tuple(n_ffts) + (1,)
                fourier_transform = FourierFeatures(n_ffts)
                bathy_transforms.append(fourier_transform)
            if op.lower() == "model":
                bmod = keras.models.load_model(args.bathy_feature_model)
                bathy_shape = tuple([int(x) for x in bmod.output.shape[1:]])
                del bmod
                bathy_model_trans = ModelFeatures(args.bathy_feature_model)
                bathy_transforms.append(bathy_model_trans)

    if len(bathy_transforms) == 0:
        bathy_transforms = None

    return [bathy_transforms, [NoOp()]]


