CRC5Docker
==========

Python scripts and Jupyter Notebooks for the textbook 
"Image Analysis, Classification and Change Detection in Remote Sensing, Fifth Revised Edition"
included in the Docker container mort/crc5docker.

Pull and/or run the container for the first time with

    sudo docker run -d -p 8888:8888 -v <path-to-crc5imagery>:/home/imagery/ --name=crc5 mort/crc5docker

This maps the host directory _crc5imagery_ to the container directory /home/imagery/ and runs the
container in detached mode. The compressed  _crc5imagery_ directory can be downloaded from

... (TBA)

Point your browser to http://localhost:8888 to see the JupyterLab home page and open a Chapter notebook.

Stop the container with

    sudo docker stop crc5  
     
Re-start with

    sudo docker start crc5     
    
The scripts are documented in 

    out/python_scripts.pdf   
