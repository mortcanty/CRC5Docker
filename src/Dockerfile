# Dockerfile with software for
# Image Analysis, Classification and Change Detection
# in Remote Sensing, Fifth Revised Edition

FROM     ubuntu:22.04

LABEL  "maintainer"="mort.canty@gmail.com"

ENV     REFRESHED_AT=2025-05-02

# last 3 installs are required for opencv
RUN apt-get update && apt-get install -y \
    curl \
    python3  \
    build-essential  \
    pandoc  \
    gdal-bin  \
    python3-gdal  \
    python3-pip \
    libgdal-dev \
    ffmpeg \
    libsm6 \
    libxext6

# jupyter notebook and assorted packages
RUN     pip3 install --upgrade pip
COPY    requirements.txt ./
RUN     pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8888

# setup the prov_means library
COPY    prov_means.c /home/prov_means.c
WORKDIR /home
RUN     gcc -shared -Wall -g -o libprov_means.so -fPIC prov_means.c
RUN     cp libprov_means.so /usr/lib/libprov_means.so
RUN     rm prov_means.c

# install auxil
ADD     auxil /home/auxil
COPY    setup.py /home/setup.py
#RUN     python3 setup.py install
RUN     pip3 install -e .

# imagery directory to be linked with local
RUN     mkdir /home/imagery

# textbook scripts, notebooks and images
ADD     scripts /home/scripts
ADD     pngs /home/pngs
COPY    Chapter?.ipynb /home/

# ipython notebook startup script
COPY    notebook.sh /
RUN     chmod u+x /notebook.sh
WORKDIR /home

# ----Experimental RAG/LLM for textbook-------
#COPY   crc5rag.ipynb /home/
#RUN    curl -fsSL https://ollama.com/install.sh | sh
#RUN    pip3 install chromadb
#RUN    pip3 install gradio
# --------------------------------------------

CMD    ["/notebook.sh"]


