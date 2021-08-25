#!/bin/bash

cd ../
docker build -f docker/Dockerfile.gpu -t habitat .
