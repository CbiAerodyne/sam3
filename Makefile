# ---------------------------------------------------------
# Custom Makefile for RTX 50-Series (Blackwell) Setup
# ---------------------------------------------------------

# 1. Image Name
# We give it a distinct name so we don't mix it up with standard versions
IMAGE_NAME := sam3:v1

# 2. Dockerfile Source
# This tells Docker to use your custom file, not the default one
DOCKERFILE := Dockerfile.sam3

# 3. Build Command
build-sam3:
	@echo "Building Custom Image for RTX 5060 Ti (CUDA 12.8 / Blackwell)..."
	@echo "Building based on Ubuntu 24.04, Pytorch 2.9.1"
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

# 4. Run Command
# Simplified run command that mounts your current folder
run-sam3:
	docker run --gpus all -it --rm \
	-p 8888:8888 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "$(PWD)":/home/appuser/workspace \
	-v /mnt/d:/mnt/d \
	-v /mnt/e:/mnt/e \
	-w /home/appuser/workspace \
	-e DISPLAY=$(DISPLAY) \
	--name=sam3_v1 \
	--ipc=host $(IMAGE_NAME) /bin/bash