from python:3.6

RUN apt update && \
    apt install -y cmake \
                   libopenmpi-dev \
                   libglu1-mesa \
                   libgl1-mesa-glx \
                   freeglut3 \
                   swig
ADD requirements.txt /
ADD SetUnityLowResolution.sh /
RUN pip install -r /requirements.txt
RUN /SetUnityLowResolution.sh
