#!/usr/bin/env python
from distutils.core import setup

setup(
    name='HabitatModelling',
    version="100",
    description='Habitat modelling using neural networks',
    author='JacksonShields',
    author_email='jacksonhshields@gmail.com',
    url='http://github.com/jacksonhshields/habitat-modelling.git',
    packages=[
        'habitat_modelling',
        'habitat_modelling.utils',
        'habitat_modelling.core',
        'habitat_modelling.datasets',
        'habitat_modelling.datasets.keras',
        'habitat_modelling.datasets.torch',
        'habitat_modelling.ml',
        'habitat_modelling.ml.keras',
        'habitat_modelling.ml.keras.models',
        'habitat_modelling.ml.keras.transforms',
        'habitat_modelling.ml.keras.callbacks',
        'habitat_modelling.ml.keras.layers',
        'habitat_modelling.ml.torch.transforms',
        'habitat_modelling.ml.torch',
        'habitat_modelling.ml.torch.models',
        'habitat_modelling.methods',
        'habitat_modelling.methods.cluster',
        'habitat_modelling.methods.feature_extraction',
        'habitat_modelling.methods.latent_mapping',
        'habitat_modelling.methods.latent_mapping.models',
    ],
    package_data={
        # 'abyss_deep_learning': ["third-party"]
    },
    scripts=[
        "applications/dive-to-coco",
        "applications/plot-csv",
        ],
)
