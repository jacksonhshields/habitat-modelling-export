#!/usr/bin/env bash

gdalwarp FK180119_Hawaii_Backscatter_WGS84_2m_Feb4.tif -r bilinear -tr 0.000027840049553 -0.000027840049553 backscatter_3m.tif