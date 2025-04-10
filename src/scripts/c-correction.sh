#!/bin/sh
#
# C-correction of a multi-spectral image
# Uses masks generated by Gaussian mixture clustering on first 3+ PCs
#
# Usage:
#
#  ./c-correction.sh spatialDims bandPos numEMClasses solarAzimuth solarElevation msImage demImage
echo 
echo "=================================================="
echo "                  C-Correction"
echo "=================================================="
echo MS image: $6
echo DEM image: $7
echo spatial subset: $1
echo spectral subset: $2
echo number of classes: $3
echo solar azimuth $4:
echo solar elevation $5:
imagePixelSize=$(gdalinfo $6 | grep Pixel | awk '{print $4}')
demPixelSize=$(gdalinfo $7 | grep Pixel | awk '{print $4}')  
echo Image pixel size: $imagePixelSize
echo DEM pixel size: $demPixelSize	   
echo 'PCA... '  
pcaImage=$(python /home/pca.py -n -d $1 -p $2 $6 | tee /dev/tty \
	   | grep written \
	   | awk '{print $4}')
dims=$(echo $1 | sed -e 's/\[[0-9]*\,[0-9]*/[0,0/')
echo 'Gaussian clustering...'
classImage=$(python /home/em.py -d $dims -p [1,2,3] -K $3 $pcaImage | tee /dev/tty \
	   | grep written \
	   | awk '{print $5}')
echo   Slope and aspect...
python /home/c_corr.py -d $1 -p $2 -c $classImage $4 $5 $6 $7 | tee /dev/tty \
	   | grep written \
	   | awk '{print $5}'