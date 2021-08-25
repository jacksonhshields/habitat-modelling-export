import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def create_image_transform_list_from_args(args):
    """
    Performs all imag
    Args:
        args: (ArgumentParser) The argument parser list given to the program. It needs the following attributes:
            - img_transforms
            - image_shape
    Returns:
        [list, TransformerClass]: preprocessing list, postprocessor

    """
    image_shape = tuple([int(x) for x in args.image_shape.split(',')])

    process_mean = False

    img_transforms = []
    for op in args.img_transforms.split(','):
        if op.lower() == "crop":
            random_crop = transforms.RandomCrop((768, 768))
            img_transforms.append(random_crop)
        elif op.lower() == "resize":
            resize_transform = transforms.Resize(tuple(image_shape[:2]))
            img_transforms.append(resize_transform)
        elif op.lower() == "imgaug":
            aug_list = [
                transforms.ColorJitter(brightness=0.25, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip()]
            img_transforms.extend(aug_list)
        elif op.lower() == "bw":
            img_transforms.append(transforms.Grayscale(num_output_channels=1))
        elif op.lower() == "float" or op.lower() == "tensor":
            float_transform = transforms.ToTensor()
            img_transforms.append(float_transform)
        elif op.lower() == "mean":
            raise NotImplementedError("TODO")
            dataset = json.load(open(args.coco_path, 'r'))
            if 'info' in dataset:
                if 'channel_means' in dataset['info']:
                    channel_means = dataset['info']['channel_means']
                    process_mean = True
    if len(img_transforms) == 0:
        img_transforms = None

    if process_mean:
        post_processor = transforms.ToPILImage()
    else:
        post_processor = transforms.ToPILImage()

    return [transforms.Compose(img_transforms), post_processor]

class UnNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for c in range(tensor.shape[0]):
            tensor[c,...] *= self.std[c]
            tensor[c,...] += self.mean[c]
        return tensor


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def image_transforms_from_cfg(cfg):
    """
    cfg is a dictionary, where the key is the type of transform, and the entry configures the transform
    """
    img_transforms = []
    post_transforms = []
    for k,v in cfg.items():
        if k == "crop":
            random_crop = transforms.RandomCrop(tuple(v['size']))
            img_transforms.append(random_crop)
        elif k == "resize":
            resize = transforms.Resize(tuple(v['size']))
            img_transforms.append(resize)
        elif k == "imgaug":
            aug_trans = []
            for t in v:
                if t == "jitter":
                    aug_trans.append(transforms.ColorJitter(brightness=0.25, contrast=0.1, saturation=0.1, hue=0.1))
                elif t == "vflip":
                    aug_trans.append(transforms.RandomVerticalFlip())
                elif t == "hflip":
                    aug_trans.append(transforms.RandomHorizontalFlip())
            img_transforms.extend(aug_trans)
        elif k == "bw":
            img_transforms.append(transforms.Grayscale(num_output_channels=1))
        elif k == "tensor":
            float_transform = transforms.ToTensor()
            img_transforms.append(float_transform)
        elif k == "normalize" or k == "normalise":
            normalize = transforms.Normalize(v['mean'], v['std'])
            img_transforms.append(normalize)
            post_transforms.append(UnNormalize(v['mean'], v['std']))
        elif k == "zero_mean":
            def zero_mean(tensor):
                for c in range(tensor.shape[0]):
                    tensor[c,...] -= tensor[c,...].mean()
                return tensor
            img_transforms.append(transforms.Lambda(zero_mean))

    post_transforms.append(transforms.ToPILImage())

    return transforms.Compose(img_transforms), transforms.Compose(post_transforms)








