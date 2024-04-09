# IdDecoder

## Basic Requirements
- Ubuntu 20.04
- Pytorch 1.7.1
- Cuda Toolkit 11.2.2

### Hardware recommendations
This framework has been successfully tested on:
- CPU: Intel core i7 10th generation
- GPU: Nvidia RTX 3090 (this is not a requirement, any GPU with at least 12GB of VRAM should be enough)
- RAM: 32 GB DDR4

### Software Requirements
- Anaconda/Mini conda
- Latest Nvidia driver

1. We recommend to set up the virtual environment by Mini Conda:

```
git clonehttps://github.com/minha12/IdDecoder.git
cd IdDecoder
conda env create -n iddecoder --file ./requirements.yaml
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```


#### Special requirements for Nvidia RTX 30 series

- Install Cuda Toolkit 11.2.2
- Pytorch 1.7.1:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

# Using Docker

**Prerequisites:**
- Docker installed on your machine.
- NVIDIA GPU with the NVIDIA Container Toolkit (for GPU support).

#### Step 1: Clone the Repository
- Clone the repository containing the Gradio app and Dockerfile to your local machine.

#### Step 2: Build the Docker Image
- Open a terminal or command prompt.
- Navigate to the directory containing the Dockerfile.
- Run the following command to build the Docker image:
```
docker build -t iddecoder_docker .
```
This command builds a Docker image named `iddecoder_docker` from the Dockerfile in the current directory.

#### Step 3: Run the Docker Container
- Once the image is built, you can run the container with the following command:
```
docker run -d --gpus all -p 7860:7860 iddecoder_docker tail -f /dev/null
```
This command runs the container in detached mode (`-d`), enables GPU access (`--gpus all`), and maps port `7860` of the container to port `7860` on your host machine.

#### Step 4: Stop the Docker Container
- If you need to stop the running Docker container, first find the container ID using:
  ```
  docker ps
  ```
- Then stop the container using the following command:
  ```
  docker stop [container_id]
  ```
  Replace `[container_id]` with the actual ID of your container.
