import numpy as np
import cv2
import PIL
from PIL import Image
import imutils
from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d
import random
from imgaug import augmenters as iaa
import cv2
import json

class ResizeImage:
    """
    Resizes an image
    """
    def __init__(self, img_shape):
        """
        Resizes an image
        Args:
            img_shape: (width,height) tuple
        """
        if len(img_shape) == 3:
            img_shape = tuple(img_shape[:2])
        self.img_shape = img_shape
    def transform(self, img):
        pimg = Image.fromarray(img)
        pimg = pimg.resize(self.img_shape)
        return np.array(pimg)



class ImageToFloat:
    """
    Changes an image from 0->255 to -1.0->1.0
    """
    def transform(self, img):
        img = img / 127.5 - 1.0
        return img


class FloatToInt:
    """
    Changes an images from -1.0->1.0 to 0->255
    """
    def transform(self, img):
        img = (img + 1) * 127.5
        img = img.astype(np.uint8)
        return img


class RandomCrop:
    """
    Grabs a crop from the image and returns it
    """
    def __init__(self, img_shape):
        """
        Grabs a random crop from the image and returns it
        Args:
            img_shape: (width,height) tuple
        """
        if len(img_shape) == 3:
            img_shape = tuple(img_shape[:2])
        self.img_shape = img_shape

    def transform(self, img):
        pimg = Image.fromarray(img)
        # If the target image shape is less than the current image shape, just resize and return
        if self.img_shape[0] > pimg.width or self.img_shape[1] > pimg.height:
            pimg = pimg.resize(self.img_shape)
            return np.array(pimg)
        else:  # Crop a part of the image and return it
            fullbbox = pimg.getbbox()  # Left, upper, right, lower
            # Point left
            pl = fullbbox[0] - self.img_shape[0]
            # Point upper
            pu = fullbbox[1] - self.img_shape[1]
            # If less than 0, set to 9
            if pl < 0:
                pl = 0
            if pu < 0:
                pu = 0
            pl = random.randint(0,pl)
            pu = random.randint(0,pu)

            cropbbox = (pl, pu, pl + self.img_shape[0], pu + self.img_shape[1])  # Add target width height to create cropped bbox
            cimg = pimg.crop(cropbbox)
            return np.array(cimg)


class ImageAugmenter:
    """
    Performs image augmentation using imgaug. Info at https://imgaug.readthedocs.io/en/latest/source/augmenters.html
    """
    def __init__(self, some_of=2, flip_lr=True, flip_ud=True, gblur=None, avgblur=None, gnoise=None, scale=None, rotate=None, bright=None, colour_shift=None):
        """
        More info at https://imgaug.readthedocs.io/en/latest/source/augmenters.html

        Args:
            some_of: (int) The maximum amount of transforms to apply. Randoms selects 0->some_of transforms to apply.
            flip_lr: (bool) Randomly flip the images from left to right
            flip_ud: (bool) randomly flip the images from left to right
            gblur: (tuple) Apply gaussian blur. This is equal to sigma. gblur=(0.0,3.0)
            avgblur: (tuple) Apply average blur. This is equal to kernel size e.g. avgblur=(2,11)
            gnoise: (tuple) Apply guassian noise. This parameter is equal to scale range. e.g. gnoise=(0,0.05*255)
            scale: (tuple) Apply scale transformations. This parameter is equal to scale. e.g. scale=(0.5,1.5) to scale between 50% and 150% of image size
            rotate: (tuple) Apply rotation transformations. This parameter is equal to rotation degrees. e.g. scale=(-45,45)
            bright: (tuple) Brighten the image by multiplying. This parameter is equal to the range of brightnesses, e.g. bright=(0.9,1.1)
            colour_shift: (tuple) Apply a color slight colour shift to some of the channels in the image. This parameter is equal to the multiplying factor on each channel. E.g. colour_shift=(0.9,1.1)

        """
        self.aug_list = []
        if flip_lr:
            self.aug_list.append(iaa.Fliplr(0.5))
        if flip_ud:
            self.aug_list.append(iaa.Flipud(0.5))
        if gblur:
            self.aug_list.append(iaa.GaussianBlur(sigma=gblur))
        if avgblur and not gblur:
            # Only use avgblur if gblur is not being used
            self.aug_list.append(iaa.AverageBlur(k=avgblur))
        if gnoise:
            self.aug_list.append(iaa.AdditiveGaussianNoise(scale=gnoise))
        if scale:
            self.aug_list.append(iaa.Affine(scale=scale))
        if rotate:
            self.aug_list.append(iaa.Affine(rotate=rotate))
        if bright:
            self.aug_list.append(iaa.Multiply(bright))
        if colour_shift:
            colours = iaa.SomeOf((0, None),[
                iaa.WithChannels(0, iaa.Multiply(mul=colour_shift)),
                iaa.WithChannels(1, iaa.Multiply(mul=colour_shift)),
                iaa.WithChannels(2, iaa.Multiply(mul=colour_shift))])
            self.aug_list.append(colours)

        self.some_of = some_of

    def transform(self, img):
        num_transforms = random.randint(0,self.some_of)
        if num_transforms == 0:
            return img
        else:
            aug = iaa.SomeOf(num_transforms, self.aug_list, random_order=True)
            img_aug = aug.augment_image(img)
            return img_aug


class MeanPreProcessorInt:
    """
    Subtracts the dataset mean from the image. Outputs an uint8 image, which can cause clipping.
    """
    def __init__(self, rMean, gMean, bMean):
        """
        Initialisation
        Args:
            rMean: the mean of red channel - range 0->255
            gMean: the mean of green channel - range 0->255
            bMean: the mean of blue channel - range 0->255
        """
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def transform(self, image):
        image = image.astype(np.float32)

        image[:,:,0] -= self.rMean
        image[:,:,1] -= self.gMean
        image[:,:,2] -= self.bMean

        return image.astype(np.uint8)

class MeanPostProcessorInt:
    """
    Subtracts the dataset mean from the image. Outputs an uint8 image, which can cause clipping.
    """
    def __init__(self, rMean, gMean, bMean):
        """
        Initialisation
        Args:
            rMean: the mean of red channel - range 0->255
            gMean: the mean of green channel - range 0->255
            bMean: the mean of blue channel - range 0->255
        """
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def transform(self, image):
        image = image.astype(np.float32)

        image[:,:,0] += self.rMean
        image[:,:,1] += self.gMean
        image[:,:,2] += self.bMean

        # return image.astype(np.uint8)
        return image


class MeanPreProcessorFloat:
    """
    Subtracts the mean of each channel. Outputs a float image range -1,1 so needs to be after ImageToFloat in augmentation list.
    """
    def __init__(self, rMean, gMean, bMean):
        """
        Initialisation. Converts mean ranges from 0->255 to -1->1
        Args:
            rMean: the mean of red channel - range 0->255
            gMean: the mean of green channel - range 0->255
            bMean: the mean of blue channel - range 0->255
        """
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean / 127.5 - 1.0
        self.gMean = gMean / 127.5 - 1.0
        self.bMean = bMean / 127.5 - 1.0

    def transform(self, image):
        image = image.astype(np.float32)

        image[:,:,0] -= self.rMean
        image[:,:,1] -= self.gMean
        image[:,:,2] -= self.bMean

        return image

class MeanPostProcessorFloat:
    """
    Adds the mean of each channel. Input range -is -1,1 so needs to be before FloatToInt in postprocessing list.
    """
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean / 127.5 - 1.0
        self.gMean = gMean / 127.5 - 1.0
        self.bMean = bMean / 127.5 - 1.0

    def transform(self, image):
        """
        Initialisation. Converts mean ranges from 0->255 to -1->1
        Args:
            rMean: the mean of red channel - range 0->255
            gMean: the mean of green channel - range 0->255
            bMean: the mean of blue channel - range 0->255
        """
        image = image.astype(np.float32)

        image[:,:,0] += self.rMean
        image[:,:,1] += self.gMean
        image[:,:,2] += self.bMean

        return image

class PostTransformerInt:
    """
    Performs all post processing for a mean int preprocessor
    """
    def __init__(self, rMean, gMean, bMean):
        """
        Initialisation
        Args:
            rMean: the mean of red channel - range 0->255
            gMean: the mean of green channel - range 0->255
            bMean: the mean of blue channel - range 0->255
        """
        self.mean_processor = MeanPostProcessorInt(rMean, gMean, bMean)
        self.int_processor = FloatToInt()
    def transform(self, image):
        for tr in [self.int_processor, self.mean_processor]:
            image = tr.transform(image)
        return image


class PostTransformerFloat:
    """
    Performs all post processing for a mean float preprocessor
    """
    def __init__(self, rMean, gMean, bMean):
        """
        Initialisation
        Args:
            rMean: the mean of red channel - range 0->255
            gMean: the mean of green channel - range 0->255
            bMean: the mean of blue channel - range 0->255
        """
        self.mean_processor = MeanPostProcessorFloat(rMean, gMean, bMean)
        self.int_processor = FloatToInt()
    def transform(self, image):
        for tr in [self.mean_processor, self.int_processor]:
            image = tr.transform(image)
        return image


class ImageToGray:
    """
    Converts the image to black and white
    """
    def transform(self, img):
        """
        Converts the image to grayscale
        Args:
            img: (np.ndarray) rgb image in format (W,H,C)

        Returns:
            np.ndarray: grayscale image in format (W,H,1)

        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=-1)  # Add the extra channel
        return gray


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
            random_crop = RandomCrop((768, 768))
            img_transforms.append(random_crop)
        elif op.lower() == "resize":
            resize_transform = ResizeImage(tuple(image_shape[:2]))
            img_transforms.append(resize_transform)
        elif op.lower() == "imgaug":
            aug_transform = ImageAugmenter(
                some_of=3,
                flip_lr=True,
                flip_ud=True,
                bright=(0.5, 1.5),
                colour_shift=(0.9,1.1)
            )
            img_transforms.append(aug_transform)
        elif op.lower() == "float":
            float_transform = ImageToFloat()
            img_transforms.append(float_transform)
        elif op.lower() == "mean":
            dataset = json.load(open(args.coco_path, 'r'))
            if 'info' in dataset:
                if 'channel_means' in dataset['info']:
                    channel_means = dataset['info']['channel_means']
                    mean_transform = MeanPreProcessorFloat(channel_means['red'], channel_means['green'],
                                                      channel_means['blue'])
                    img_transforms.append(mean_transform)
                    process_mean = True
    if len(img_transforms) == 0:
        img_transforms = None

    if process_mean:
        post_processor = PostTransformerFloat(channel_means['red'], channel_means['green'], channel_means['blue'])
    else:
        post_processor = FloatToInt()

    return [img_transforms, post_processor]
