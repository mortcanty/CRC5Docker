#!/bin/sh
ollama serve &
sleep 5
jupyter lab --no-browser --port 8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
