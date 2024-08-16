# Dockerfile with software for
# Image Analysis, Classification and Change Detection
# in Remote Sensing, Fifth Revised Edition
 
FROM     debian:bullseye

LABEL  "maintainer"="mort.canty@gmail.com"

ENV     REFRESHED_AT=2024-06-26

RUN apt-get update && apt-get install -y \
    python3  \
    build-essential  \
    pandoc  \
    gdal-bin  \
    python3-gdal  \
    python3-pip

# jupyter notebook and assorted packages
RUN     pip3 install --upgrade pip && \
        pip3 install jupyter && \
        pip3 install ipyleaflet && \
        pip3 install scikit-learn && \
        pip3 install spectral && \
        pip3 install -U cython && \
        pip3 install matplotlib && \
        pip3 install geopy  && \
        pip3 install shapely && \
        pip3 install geemap && \
        pip3 install opencv-python && \
        pip3 install ipyparallel && \
        pip3 install earthengine-api --upgrade && \
        pip3 install tensorflow

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

RUN     mkdir /home/imagery

# textbook scripts, notebooks and images
ADD     scripts /home/scripts
#ADD     imagery /home/imagery
ADD     pngs /home/pngs
COPY    Chapter?.ipynb /home/

# ipython notebook startup script
COPY    notebook.sh /
RUN     chmod u+x /notebook.sh

WORKDIR /home  
CMD     ["/notebook.sh"]
