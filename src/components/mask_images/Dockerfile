FROM --platform=linux/amd64 python:3.8-slim

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]