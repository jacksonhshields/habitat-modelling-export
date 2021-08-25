# Habitat Modelling

This repository is dedicated to exploring habitat modelling and adaptive sampling with deep neural networks.


## Installation


### Requirements
This package is run with and tested with python 3.6.

It is designed to run with either keras or pytorch. Only one of the two should need to be installed.

For this package to be run, the following python packages are needed (install via pip):
- tensorflow (tested with 1.13
- keras (tested with 2.2.4)
- pytorch (tested with 1.1.0)
- numpy
- matplotlib
- neptune-client
- shapely
- pyproj
- utm
- pymap3d
- gdal
- opencv
- Pillow
- pycocotools

### Installation

Install use pip:
```bash
git clone https://github.com/jacksonhshields/habitat-modelling.git
cd habitat-modelling
pip3 install --user --upgrade .
```


## Dataset Creation

### Extended COCO Dataset Format

The datasets used for training are a COCO format json file. These datasets are flexible, informative and can easily be used for
active learning (easy to create from inference application). The coco format is defined [here](http://cocodataset.org/#format-data).
This uses an extension of the COCO interface (but is still able to use pycocotools), by adding additional meta information to the image entries.

The following information is stored in each image entry:
```json
{
  "id": 0,
  "file_name": "image_filename.png",
  "path": "path/to/image_filename.png",
  "height": 1024,
  "width": 1360,
  "pose": {
    "altitude": 2.123,
    "orientation": [
      0.0069012589455976,
      0.0109577111106051,
      -3.109733292633899
    ],
    "position": [
      37.986901427498644,
      88.30840074422372,
      36.34583007183288
    ]
  },
  "geo_location": [
    -43.084215926116094,
    147.97577091051235,
    -36.34583007183288
  ]
}
```
Similarly, an entry for each rater needs to be given, for example a bathymetry raster:
```json
{
  "bathymetry": {
    "path": "/path/to/bathymetry.tif" 
  }
}
```


### Creating the COCO dataset from ACFR Dive

The coco json file can be created directly from an ACFR dive, using the 'dive-to-coco' application:
```bash
DIVE_PATH="/media/water/processed/2008/Tasmania200810/r20081014_052323_ohara_20_oneline"
RENAV_SUBDIR="renav20160205"
OUTPUT_DIR="/media/water/datasets/2008/Tasmania200810/r20081014_052323_ohara_20_oneline/"
OUTPUT_COCO="$OUTPUT_DIR/ohara_20_bathy.json"
RENAV_POSE_FILE="/media/water/processed/2008/Tasmania200810/r20081014_052323_ohara_20_oneline/renav20160205/stereo_pose_est.data"
BATHYMETRY_TIF="/media/water/Tassie_bathy/Tasmania200810/TAFI_provided_data/BathymetryAsTiffs/fort1.tif"
TRIP="Tasmania200810"
YEAR="2008"

~/src/habitat-modelling/applications/dive-to-coco \
    $DIVE_PATH \
    $OUTPUT_COCO \
    --renav-pose-file $RENAV_POSE_FILE \
    --bathymetry $BATHYMETRY_TIF \
    --trip $TRIP \
    --year $YEAR

```
The script above creates a coco json file, that can be loaded directly into the generators (either pytorch or keras) and used for training, inference etc.


### Loading the COCO dataset (Keras)

The COCO dataset can be used as input to a Generator, that will load the dataset on the fly (and use multiprocessing etc.). This is done by using a generator that is a subclass of ([keras.utils.Sequence](https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L305)).

The custom keras generator can be found [here](https://github.com/jacksonhshields/habitat-modelling/blob/master/habitat_modelling/datasets/keras/coco_habitat.py), and can be called by:
```python
    from habitat_modelling.datasets.keras.coco_habitat import CocoImgBathyGenerator
    train_generator = CocoImgBathyGenerator(coco_path, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            img_transforms=img_transforms,  # A list of preprocessing transforms for the images - classes that have a 'transform' function. 
                                            bathy_transforms=bathy_transforms)  # A list of preprocessing transforms for bathymetry - classes that a 'transform' function.
```
There are two options for using the generator, either using 'fit_generator' or 'enumerate'
```python
model.fit_generator(train_generator, epochs=10)
```
```python
for epoch in range(num_epochs):
    for batch, (img_batch, dmap_batch, depth_batch) in enumerate(train_generator):
        loss = model.train_on_batch(img_batch, img_batch)  # For an autoencoder
```

### Loading the COCO dataset (PyTorch)

The custom PyTorch generator can be found [here](https://github.com/jacksonhshields/habitat-modelling/blob/master/habitat_modelling/datasets/torch/coco_habitat.py), and can be called by:

```python
from habitat_modelling.datasets.torch.coco_habitat import CocoImgBathy
import torch.utils.data
import torchvision.transforms as transforms

dataset = CocoImgBathy(dataset_path,
                        img_transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]),  # PyTorch transforms
                        dmap_transform=transforms.Compose([transforms.ToTensor()]),
                        depth_transform=transforms.Compose([transforms.ToTensor()])
                      )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

```

The model can then be trained by using 'enumerate':
```python
for epoch in range(num_epochs):
    for batch, (imgs, dmaps, depths) in enumerate(dataloader):
        optimizer.zero_grad()
        imgs = Variable(imgs.type(Tensor))
        h = encoder(imgs)
        x = decoder(h)
        r_loss = reconstruction_loss(x, imgs)
        r_loss.backward()
        optimizer.step()
```


## Creating new experiments

Use the experiments directory for experimenting with new models, training procedures etc.

## Extras

### coco-mapper
[coco-mapper](https://github.com/jacksonhshields/habitat-modelling/blob/master/applications/coco-mapper.py) is an application used to create kml files of the images and their associated classes (e.g. from an inference run). 
An example of using this script is:
```bash
python3 ~/src/habitat-modelling/applications/coco-mapper.py /path/to/inference-coco.json /path/to/inference.kml
```
This can then be loaded into Google Earth Pro and visualised.

### plot-csv
The training scripts output CSV files containing the training metrics etc. This application allows a quick and easy plotting
of any CSV file. For example, to plot the image to image and bathy to bathy loss together:
```bash
plot-csv /path/to/training_logs.csv --y-axis 'i2i Loss,b2b Loss'
```


