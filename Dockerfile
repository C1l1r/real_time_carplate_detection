FROM ubuntu:20.04

RUN apt-get update

# Install Python 3.8
RUN apt-get install -y \
    python3 \
    python3-dev \
    python3-pip

# Install PyTorch 1.13
RUN pip3 install torch



# Copy  and install other requirements
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY requirements.txt /app
ADD sort /app/
COPY models/best.pt /app/models/
COPY *.py /app/
COPY yolov8n.pt /app/
COPY models /app/models
RUN pip3 install -r ./requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y


# final configuration
CMD python3 main.py
