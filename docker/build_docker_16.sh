#!/bin/bash

cd ../
docker build -f docker/Dockerfile_16.gpu -t habitat16 .
