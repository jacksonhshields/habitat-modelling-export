#!/bin/bash

cd ../
docker build -f docker/Dockerfile_16_p27_acfr.gpu -t habitat16p27 .
