FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y && \
    apt-get install gcc -y && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]