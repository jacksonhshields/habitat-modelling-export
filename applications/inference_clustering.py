#!/usr/bin/env python3
import os
import keras
import numpy as np
from pycocotools.coco import COCO
import argparse
from habitat_modelling.ml.keras.transforms.image_transforms import FloatToInt, ResizeImage, ImageToFloat, RandomCrop, ImageAugmenter
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from keras.models import Model
from PIL import Image
import pickle
import json

def get_args():
    parser = argparse.ArgumentParser(description="Creates a coco json from an ACFR Marine Dive")
    parser.add_argument('coco_path', help="Path to the coco format training dataset")
    parser.add_argument('output_coco', help="Path to the coco format training dataset")
    parser.add_argument('--model-path', help="Path to model keras model")
    parser.add_argument('--load-pickle', help="Path to load the pickle file from (saves running inference again)")
    parser.add_argument('--save-pickle', help="Path to save the pickle file to")
    parser.add_argument('--input-layer', help="The input layer for the network, e.g. 'img_input'")
    parser.add_argument('--output-layer', help="The output layer for the network, e.g. 'global_average_pooling2d_1'")
    parser.add_argument('--preprocessing-transforms', default='resize,float', help='The preprocessing transforms to apply to the images')
    parser.add_argument('--post-processing-transforms', default='int', help='The post-processing transforms to apply')
    parser.add_argument('--image-shape', type=str, default="256,256,3", help="Image Shape in form \"width, height, channels\"")
    parser.add_argument('--crop-shape', type=str, default="768,768", help="Random Crop image hape in form \"width, height\"")
    parser.add_argument('--image-dir', type=str, help="Optionally set the directory of the images")
    parser.add_argument('--tsne', type=int, help="Use tsne - value indicates number of dimensions to compress to")
    parser.add_argument('--clustering', type=str, help="The clustering method to use. Options {kmeans,dbscan}")
    parser.add_argument('--num-clusters', type=int, default=10, help="The number of clusters to be used")
    parser.add_argument('--plot', action="store_true", help="Whether to plot the results")
    parser.add_argument('--neptune', type=str, help="The neptune experiment name. Omitting this means neptune isn't used.")

    args = parser.parse_args()
    return args


def cut_model(model, input_layer, output_layer):
    """
    Creates a new model with a different input/output layer.

    Only works for a 1 input, 1 output scenario.. TODO make it work for multi input

    Args:
        model: the original model
        input_layer: the input layer, e.g. img_input
        output_layer: the output layer, e.g. global_average_pooling2d_1

    Returns:
        Model: the new, cut down, model
    """
    if input_layer and output_layer:
        model = Model(inputs=model.get_layer(input_layer).input,
                      outputs=model.get_layer(output_layer).output)
    elif input_layer and not output_layer:
        model = Model(inputs=model.get_layer(input_layer).input,
                      outputs=model.output)
    elif output_layer and not input_layer:
        model = Model(inputs=model.input,
                      outputs=model.get_layer(output_layer).output)
    return model

def predict_output(args):
    model = keras.models.load_model(args.model_path)

    model = cut_model(model, args.input_layer, args.output_layer)

    image_shape = tuple([int(x) for x in args.image_shape.split(',')])

    crop_shape = tuple([int(x) for x in args.crop_shape.split(',')])

    coco = COCO(args.coco_path)

    # Create the image transforms
    preprocessing_transforms = []
    for op in args.preprocessing_transforms.split(','):
        if op.lower() == "crop":
            random_crop = RandomCrop(crop_shape)
            preprocessing_transforms.append(random_crop)
        elif op.lower() == "resize":
            resize_transform = ResizeImage(tuple(image_shape[:2]))
            preprocessing_transforms.append(resize_transform)
        elif op.lower() == "imgaug":
            aug_transform = ImageAugmenter()
            preprocessing_transforms.append(aug_transform)
        elif op.lower() == "float":
            float_transform = ImageToFloat()
            preprocessing_transforms.append(float_transform)

    post_processing_transforms = []
    for op in args.post_processing_transforms.split(','):
        if op.lower() == "int":
            post_processing_transforms.append(FloatToInt())

    total_images = len(coco.getImgIds())

    # ID list and output list are matched - each appended to at the same time and should have same length.
    # Used to add the annotations after
    id_list = []
    output_list = []
    for i, image in enumerate(coco.loadImgs(coco.getImgIds())):
        # Load the image
        if args.image_dir:
            img_path = os.path.join(args.image_dir, image['file_name'])
        elif 'path' in image:
            img_path = image['path']
        else:
            img_path = image['file_name']

        img = np.array(Image.open(img_path))

        for transformer in preprocessing_transforms:
            img = transformer.transform(img)

        output = model.predict_on_batch(np.expand_dims(img, axis=0))  # predict on a batch
        output = np.squeeze(output, axis=0)  # Remove the batch element

        output_list.append(output)  # Add the outputs to a list, will be converted to array after
        id_list.append(image['id'])  # Add the ids to a list, to mimic the output

        print("Inferring image %i/%i" %(i, total_images))

    X = np.asarray(output_list)
    id_array = np.asarray(id_list)

    if args.save_pickle:
        data = {"output": X, "id": id_array}
        pickle.dump(data, open(args.save_pickle, 'wb'))

    return X, id_array

def load_output(args):
    data = pickle.load(open(args.load_pickle, 'rb'))
    X = data['output']
    id_array = data['id']
    return X, id_array


def main(args):
    if not args.load_pickle and not args.model_path:
        raise ValueError("Either a model path needs to be given or a pickled output from a previous inference")

    if args.load_pickle:
        X, id_array = load_output(args)
    else:
        X, id_array = predict_output(args)


    if args.tsne:
        X = TSNE(args.tsne).fit_transform(X)

    X = StandardScaler().fit_transform(X)

    if args.clustering.lower() == "kmeans":
        clustering = KMeans(n_clusters=args.num_clusters)
        y_pred = clustering.fit_predict(X)
    elif args.clustering.lower() == "dbscan":
        y_pred = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    elif args.clustering.lower() == "gmm":
        clustering = GaussianMixture(n_components=args.num_clusters)
        y_pred = clustering.fit_predict(X)
    else:
        y_pred = None
        raise ValueError("Invalid clustering method")

    if len(id_array) != X.shape[0]:
        raise ValueError("Length of IDs need to be equal to length of output array")

    # Create category list
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise_ = list(y_pred).count(-1)  # Leave noise as empty annotations
    cats = []
    for cl in range(n_clusters):
        cat = {
            "id": cl,
            "name": "cluster_%d" %cl,
            "supercategory": ""
        }
        cats.append(cat)



    anns = []
    for n in range(len(id_array)):
        if y_pred[n] != -1:
            ann = {
                "id": n,
                "annotation_type": "point",
                "image_id": int(id_array[n]),
                "category_id": int(y_pred[n]),
                "area": 10, # These need to be there for annotation tool
                "bbox": [1,2,3,4],
                "iscrowd": 0,
                "occluded": False,
                "segmentation": [1,2,3,4]
            }
            anns.append(ann)

    output_coco = copy.deepcopy(COCO(args.coco_path).dataset)
    output_coco['annotations'] = anns
    output_coco['categories'] = cats

    json.dump(output_coco, open(args.output_coco, 'w'))


    if args.plot:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.show()

if __name__ == "__main__":
    main(get_args())