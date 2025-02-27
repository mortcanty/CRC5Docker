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

https://drive.google.com/file/d/1zLJke10J0Otr09I0LYmF5gJ7MIzpZgct/view?usp=sharing

Point your browser to http://localhost:8888 to see the JupyterLab home page and open a Chapter notebook.

Stop the container with

    docker stop crc5  
     
Re-start with

    docker start crc5     

For LLM enthusiasts an experimental RAG (retrieval augmented generation) 
version of the Docker container can be pulled and run with

    docker run -d -p 8888:8888 -p 7860-7869:7860-7869 
        -v <path-to-crc5imagery>:/home/imagery/ --name=crc5_rag
            mort/crc5docker_rag

which includes an additional JupyterLab notebook to query the 
textbook's content informally. At least 24GB RAM is required.  
Since the LLM is running locally on a CPU, the response time is very slow (minutes).
While the model usually does its best, the retrieved content is limited by
the LLM prompt window so the generated answers are often misleading. 

__Book Summary__

    chapter_abstracts.pdf

