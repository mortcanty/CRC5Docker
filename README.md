CRC5Docker
==========

Python scripts and Jupyter Notebooks for the textbook
Image Analysis, Classification and Change Detection in Remote Sensing, Fifth Revised Edition
included in the Docker image

    mort/crc5docker

The scripts are documented in 

    python_scripts.pdf

Pull and/or run the container for the first time with

    docker run -d -p 8888:8888 -v <path-to-crc5imagery>:/home/imagery/ --name=crc5 mort/crc5docker

This maps the host directory _crc5imagery_ to the container directory /home/imagery/ and runs the
container in detached mode. The compressed  _crc5imagery_ directory can be downloaded from

https://drive.google.com/file/d/1EOJolX0Diumo0ebM6xDvCZQqb34s8Gxz/view?usp=sharing

Point your browser to http://localhost:8888 to see the JupyterLab home page and open a Chapter notebook.

Stop the container with

    docker stop crc5  
     
Re-start with

    docker start crc5     

In order to be able to access the Google Earth Engine from within the Chapter notebooks you can register a free, non-commercial Google Cloud project if you have not already done so:

https://earthengine.google.com/


For LLM enthusiasts an experimental RAG (retrieval augmented generation) 
version of the Docker container can be pulled and run with

    docker run -d -p 8888:8888 -p 7860-7869:7860-7869 -v <path-to-crc5imagery>:/home/imagery/ --name=crc5_rag mort/crc5docker_rag

which includes an additional JupyterLab notebook to query the textbook's content informally. 
If the LLM is running locally on a CPU, the response time is very slow (minutes). 
and the generated answers are often misleading. If you use a cloud version (see the RAG Notebook), 
response time is in seconds and the answers are much more reliable and pertinent. A (free) ollama 
account is required.

__Book Summary__

    chapter_abstracts.pdf

__Additional Earth Engine Tutorials__

https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt1

https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt2

https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt3

https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1

https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-2

https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-3

https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-4

