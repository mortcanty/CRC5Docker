# CRC5Docker

Companion Python scripts, Jupyter notebooks and Docker images for the textbook

> **Image Analysis, Classification and Change Detection in Remote Sensing**
> Fifth Revised Edition
> Mort Canty

All material in this repository is pre-installed in the Docker image
[`mort/crc5docker`](https://hub.docker.com/r/mort/crc5docker) so that the
chapter examples and exercises can be reproduced without any local Python
setup.

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Using the Notebooks](#using-the-notebooks)
- [Google Earth Engine Setup](#google-earth-engine-setup)
- [RAG / LLM Variant](#rag--llm-variant)
- [Supplied Documentation](#supplied-documentation)
- [Additional Resources](#additional-resources)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Author & Contact](#author--contact)

---

## Overview

This repository provides the software that accompanies the fifth revised
edition of the textbook *Image Analysis, Classification and Change Detection
in Remote Sensing*. It contains:

* nine **Jupyter notebooks** (`Chapter1.ipynb` ... `Chapter9.ipynb`) with the
  worked examples from the book,
* a `src/scripts/` directory of **stand-alone Python modules** that the
  notebooks import (e.g. `iMad.py`, `kmeans.py`, `sar_seqQ.py`),
* two pre-built **Docker images** that bundle JupyterLab, TensorFlow, the
  Google Earth Engine Python API, `geemap`, `geopandas`, GDAL and the rest
  of the scientific Python stack,
* a `mort/crc5docker_rag` image variant that adds an experimental
  retrieval-augmented-generation (RAG) notebook for querying the textbook
  with a local LLM.

A supplementary ~60-page monograph on SAR change detection with Sentinel-1
(`gee-s1.pdf`) is included in the image.

## Features

* **Reproducible environment** — one `docker run` command gets you a fully
  working JupyterLab.
* **Google Earth Engine** integration out of the box, with example notebooks
  for GEE-based change detection.
* **Classical and ML methods** — scripts for PCA, MNF, ICA, iMad, k-means,
  EM, AdaBoost, RX, KRX, KPCA, RBF, SOM, mean-shift, etc.
* **SAR time-series change detection** — `sar_seqQ.py` and friends reproduce
  the sequential algorithms from the SAR monograph.
* **Experimental RAG/LLM** — query the textbook in natural language using a
  local llama3.1 model or a free ollama cloud account.

## Requirements

| Component             | Minimum                              | Recommended                |
|-----------------------|--------------------------------------|----------------------------|
| Docker Engine         | 20.10+                               | 24+                        |
| Host OS               | Linux, macOS 11+, Windows 10/11 WSL2 | Linux                      |
| Free disk space       | 8 GB                                 | 15 GB (RAG image is larger)|
| RAM                   | 8 GB                                 | 16 GB                      |
| Google Earth Engine   | required for the GEE notebooks       | —                          |
| ollama account (free) | required for the RAG cloud backend   | —                          |

## Quick Start

### 1. Download the imagery bundle

The compressed `crc5imagery` directory used by the notebooks is hosted on
Google Drive:

<https://drive.google.com/file/d/1EOJolX0Diumo0ebM6xDvCZQqb34s8Gxz/view?usp=sharing>

Unpack it somewhere on the host machine and note the path. The notebooks
expect to find it at `/home/imagery/` inside the container.

### 2. Pull and run the **base** image

```bash
docker run -d \
    -p 8888:8888 \
    -v <path-to-crc5imagery>:/home/imagery/ \
    --name=crc5 \
    mort/crc5docker
```

This command

* maps port `8888` of the container to `8888` on the host,
* bind-mounts the local `crc5imagery` directory onto `/home/imagery/` in the
  container, and
* starts the container in detached mode under the name `crc5`.

Open <http://localhost:8888> in your browser — JupyterLab will appear.
Pick a `ChapterN.ipynb` notebook to begin.

### 3. Stop / restart the container

```bash
docker stop  crc5     # shut down
docker start crc5     # bring it back up
```

### 4. (Optional) Run the **RAG** variant

If you also want the LLM-augmented query notebook:

```bash
docker run -d \
    -p 8888:8888 \
    -p 7860-7869:7860-7869 \
    -v <path-to-crc5imagery>:/home/imagery/ \
    --name=crc5_rag \
    mort/crc5docker_rag
```

The extra port range `7860-7869` is for the gradio web interface that serves
the RAG chat.

## Repository Layout

```
.
|-- LICENSE.txt              # MIT licence
|-- README.md                # this file
|-- run_jupyter              # helper script for running Jupyter locally
|-- chapter_abstracts.pdf    # one-page summary of every chapter
|-- errata5.pdf              # known errata for the 5th edition
|-- gee-s1.pdf               # SAR change-detection monograph (Sentinel-1)
|-- longbeach.pdf            # supporting reference for the chapter examples
|-- python_scripts.pdf       # auto-generated reference for src/scripts/*.py
|-- solutions.pdf            # worked solutions to the exercises
'-- src/
    |-- Chapter1.ipynb ... Chapter9.ipynb  # the nine chapter notebooks
    |-- crc5rag.ipynb          # RAG/LLM query notebook (gradio UI)
    |-- crc5rag_dev.ipynb      # developer version of the RAG notebook
    |-- Dockerfile             # build for mort/crc5docker
    |-- Dockerfile_rag         # build for mort/crc5docker_rag
    |-- requirements.txt       # pinned Python dependencies
    |-- setup.py               # installs the bundled `auxil` package
    |-- notebook.sh            # container entry-point
    |-- auxil/                 # helper Python package
    |-- scripts/               # ~40 stand-alone Python modules
    |-- pngs/                  # illustrations used by the notebooks
    |-- imagery/               # placeholder; real data lives in the mounted volume
    |-- build/                 # build artefacts
    '-- pythonfiles.zip        # zipped copy of scripts/ for redistribution
```

## Using the Notebooks

* The nine `ChapterN.ipynb` notebooks are best opened in order — each builds
  on the previous one.
* The `crc5rag.ipynb` notebook contains the gradio interface for asking
  natural-language questions about the textbook.
* All paths inside the notebooks assume the imagery volume is mounted at
  `/home/imagery/`. If you mount it elsewhere, edit the `IMAGERY` path near
  the top of each notebook.

## Google Earth Engine Setup

Several notebooks initialise the Earth Engine Python API with

```python
import ee
ee.Initialize(project='your-project-name')
```

Before that line will work, you must:

1. Register a free, non-commercial Google Cloud project at
   <https://earthengine.google.com/> (a Google account is required).
2. Run `earthengine authenticate` once in a terminal inside the container —
   this stores a token in `/root/.config/earthengine/`.
3. Replace `your-project-name` with the project ID you registered.

## RAG / LLM Variant

The `mort/crc5docker_rag` image is **experimental**. It includes
`crc5rag.ipynb`, which:

1. loads the textbook PDF and the chapter abstracts,
2. builds a Chroma vector index, and
3. exposes a gradio chat UI on port `7860`.

**Performance expectations**

| Backend                    | Typical answer time | Answer quality          |
|----------------------------|---------------------|-------------------------|
| Local llama3.1 on CPU      | minutes             | often misleading        |
| ollama cloud (free account)| seconds             | reliable and pertinent  |

A free ollama account is required for the cloud backend; the RAG notebook
itself walks you through signing in.

## Supplied Documentation

| File                  | Purpose                                                       |
|-----------------------|---------------------------------------------------------------|
| `python_scripts.pdf`  | reference for every module in `src/scripts/`                  |
| `chapter_abstracts.pdf` | one-paragraph summary of each chapter                       |
| `errata5.pdf`         | corrections to the 5th edition                                |
| `gee-s1.pdf`          | ~60-page SAR change-detection monograph                      |
| `longbeach.pdf`       | background reading for the example imagery                    |
| `solutions.pdf`       | worked solutions to the end-of-chapter exercises              |

## Additional Resources

Earth Engine community tutorials referenced by the notebooks:

* [iMad tutorial — part 1](https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt1)
* [iMad tutorial — part 2](https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt2)
* [iMad tutorial — part 3](https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt3)
* [Detecting changes in Sentinel-1 imagery — part 1](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1)
* [Detecting changes in Sentinel-1 imagery — part 2](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-2)
* [Detecting changes in Sentinel-1 imagery — part 3](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-3)
* [Detecting changes in Sentinel-1 imagery — part 4](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-4)

## Troubleshooting

* **`port is already allocated`** — another container or process is using
  `8888` on the host. Change the host side: `-p 8889:8888`, then browse to
  `http://localhost:8889`.
* **`docker: name already in use`** — a container called `crc5` (or
  `crc5_rag`) already exists. Remove it with `docker rm crc5`, or pick a
  different name with `--name=...`.
* **`ee.Initialize()` fails** — run `earthengine authenticate` inside the
  container and make sure your GCP project ID is correct.
* **RAG notebook is very slow** — that is expected on a CPU-only host.
  Switch to a free ollama cloud account for second-scale answers.
* **Out-of-disk** — the RAG image plus a downloaded LLM can consume 15+ GB.
  Free up space with `docker system prune`.

## License

This project is released under the **MIT License**. See
[`LICENSE.txt`](LICENSE.txt) for the full text.

```
MIT License — Copyright (c) 2024 Mort Canty
```

## Author & Contact

**Mort Canty**
E-mail: <mort.canty@gmail.com>

Suggestions, corrections and pull requests are welcome.

---

*Last refreshed 2026-05-12 (matches the `REFRESHED_AT` in `src/Dockerfile`).*
