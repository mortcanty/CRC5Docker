CRC5Docker
==========

Python scripts and Jupyter Notebooks for the textbook
__Image Analysis, Classification and Change Detection in Remote Sensing, Fifth Revised Edition__
included in the Docker image

    mort/crc5docker.

The scripts are documented in 

    /python_scripts.pdf   

and the book chapters are summarized in
 
    /chapter_abstracts.pdf

Pull and/or run the container for the first time with

    docker run -d -p 8888:8888 -v <path-to-crc5imagery>:/home/imagery/ --name=crc5 mort/crc5docker

This maps the host directory _crc5imagery_ to the container directory /home/imagery/ and runs the
container in detached mode. The compressed  _crc5imagery_ directory can be downloaded from

https://drive.google.com/file/d/1Ca-rs1Om2bnF79YGWnqijW7PRsZ4vKG5/view?usp=sharing

Point your browser to http://localhost:8888 to see the JupyterLab home page and open a Chapter notebook.

Stop the container with

    docker stop crc5  
     
Re-start with

    docker start crc5     
    

__Book Summary__

Chapter 1 Images, Arrays and Matrices

There are many  Earth observation satellite-based sensors, both active and passive, currently in orbit or planned for the near future.
Laying the mathematical foundation for the image analysis procedures and algorithms forming the substance of the text,
Chapter 1 begins with a short description of typical remote sensing imagery in the  optical/infrared  and synthetic aperture 
radar categories, together with their representations as digital arrays.  The multispectral ASTER system and the 
TerraSAR-X synthetic aperture radar satellite are chosen as illustrations.  Then some basic concepts of linear
algebra of vectors and matrices are introduced, namely linear dependence,  eigenvalues and eigenvectors, 
singular value  decomposition and finding minima and maxima using Lagrange multipliers. 
The latter is illustrated with the principal components analysis of a multispectral image.

Chapter 2 Image Statistics

In an optical/infrared or a synthetic aperture radar image, a given pixel value $g(i,j)$, derived 
from the measured radiation field at a satellite sensor, is never exactly reproducible. It is the 
outcome of a complex measurement influenced by instrument noise, atmospheric conditions, changing 
illumination and so forth. It may be assumed, however, that there is an underlying random mechanism 
with an associated probability distribution which restricts the possible outcomes in some way. 
Each time we make an observation, we are sampling from that probability distribution or, put another 
way, we are observing a different possible {\it realization} of the random mechanism. In Chapter 2, 
some basic statistical concepts for multi-spectral and SAR images viewed as random mechanisms are introduced.

Chapter 3 Transformations 

In the first two Chapters,  multispectral and polarimetric SAR images are represented as three-dimension\-al arrays 
of  pixel intensities (columns $\times$ rows $\times$ bands) corresponding, more or less directly, to measured radiances. 
Chapter 3 deals with other, more abstract representations which are useful in image interpretation and analysis and which  
play an important role in later Chapters. The discrete Fourier and wavelet transforms, treated, in Sections 3.1 and 
3.2 convert the pixel values in a given spectral band to linear combinations of orthogonal functions of spatial frequency 
and distance.  They may therefore be classified as spatial transformations. The principal components, minimum noise 
fraction and maximum autocorrelation factor transformations (Sections 3.3 to 3.5), on the other hand, create at each pixel location new linear 
combinations of the pixel intensities from all of the spectral bands and can properly be called spectral transformations.
    
Chapter 4 Filters, Kernels and Fields

Chapter 4 is intended  to consolidate and extend material presented in the preceding three Chapters and to help  lay  the 
foundation for the rest of the book. In Sections 4.1 and 4.2, building on the discrete Fourier transform introduced in 
Chapter 3, the concept of discrete convolution  is introduced and filtering, both in the spatial and in the frequency domain, 
is discussed. Frequent reference to filtering will be made in Chapter 5 when  enhancement and geometric and radiometric 
correction of visual/infrared and SAR imagery are treated and in the discussion of convolutional neural networks in Chapter 6. 
In Section 4.3 it is shown that the discrete wavelet transform of Chapter 3 is equivalent to a
recursive application of low- and high-pass filters (a filter bank) and a pyramid algorithm for multi-scale image 
representation is described and programmed in Python. Wavelet pyramid representations are applied in Chapter 5 for
panchromatic sharpening and in Chapter 8 for contextual
clustering. Section 4.4 introduces  so-called kernelization, in which the dual representations of linear 
problems described in Chapters 2 and 3
can be modified to treat non-linear data. Kernel methods are illustrated with a non-linear version of the principal 
components transformation and they will be met again in Chapter 6 when  support vector machines for supervised 
classification are discussed, in Chapter 7 in connection with anomaly detection, and in Chapter 8 in the form of 
a kernel K-means clustering algorithm. Finally, Section 4.5 describes Gibbs--Markov random fields which are invoked 
in Chapter 8 in order to include spatial context in unsupervised  classification.

Chapter 5 Image Enhancement and Correction

In preparation for the treatment of supervised/unsupervised
classification and change detection, the
subjects of the final four chapters of this book, Chapter 5
focuses on preprocessing methods. These fall into the two general
categories of image enhancement  (Sections 5.1 through 5.4) and
geometric correction (Sections 5.5 and 5.6). Discussion mainly focuses
on the processing of optical/infrared image data. However, Section 5.4
introduces polarimetric SAR imagery and treats the  problem of
speckle removal.

Chapter 6 Supervised Classification Part 1

Land cover classification of remote sensing imagery is an undertaking which
falls into the general category of pattern recognition.
Pattern recognition problems, in turn, are usually approached by developing
appropriate  machine learning algorithms.
Broadly speaking, machine learning involves tasks for which there
is no known direct, analytic method to compute a desired output from a set
of inputs. The strategy adopted is for the computer to learn
from a set of representative examples.
Chapter 6 focuses on the case of supervised classification, which can often be seen as the modeling of probability 
distributions of the training data. On the basis of
representative data for, say,   $K$ land cover classes presumed to be present in a scene, 
the a posteriori probabilities for class $k$ conditional on observation 
g, namely pr(k|g) , $k=1 ... K$, are learned or approximated. This
is usually called the  training phase}of the classification
procedure. Then these probabilities are used to classify all of
the pixels in the image, a step  referred to as the  generalization or  prediction phase.

Chapter 7 Supervised Classification Part 2

Continuing on the subject of supervised
classification, Chapter 7  begins with a discussion  of
post classification processing methods to
improve  results on the basis of contextual
information, after which  attention is turned to statistical
procedures for evaluating classification accuracy and for making
quantitative comparisons between different classifiers. 
As examples of  ensembles of
classifiers, the adaptive boosting
technique is examined, applying it in particular to improve the generalization
accuracy of neural networks, and  the random forest classifier, an ensemble of 
binary  decision trees is also described. The remainder of the Chapter examines 
more specialized forms of supervised image classification, namely as applied to 
polarimetric SAR imagery, to data with  hyper-spectral resolution, and to intermediate and high 
resolution multispectral imagery using convolutional neural networks,  transfer learning and semantic segmentation.

Chapter 8 Unsupervised Classification

Supervised classification of remote sensing imagery,
the subject of the preceding two Chapters, involves
the use of a training dataset consisting of labeled pixels
representative of each land cover category of interest in an image. The choice of
training areas which adequately represent the spectral
characteristics of each category is very important for supervised
classification, as the quality of the training set has a profound
effect on the validity of the  result. Finding and verifying
training areas can be laborious, since the analyst must select
representative pixels for each of the classes by visual
examination of the image  and by information extraction from
additional sources such as ground reference data (ground truth),
aerial photos or existing maps.
The subject of Chapter 8, unsupervised
classification or  clustering,
requires no reference information at
all.  Instead, the attempt is made to find an underlying
class structure automatically by organizing the data into groups
sharing similar (e.g.,  spectrally homogeneous) characteristics.
Often, one only needs to specify beforehand the number K of
classes present. Unsupervised classification plays an especially important role
when very little  a priori information about the data is
available. A primary objective of using clustering algorithms
for multispectral remote sensing data  is often to
obtain useful information for the selection of training regions
in a subsequent supervised classification.

Chapter 9 Change Detection

When comparing multispectral images of a given scene taken
at different times, it is  desirable to correct the pixel
intensities as much as possible for uninteresting differences such
as those due to solar illumination, atmospheric conditions, viewing angle, terrain effects or sensor
calibration. In the case of SAR imagery, solar illumination or cloud cover play no role, but other 
considerations are similarly important.
If comparison is on a pixel-by-pixel basis, then the
images must also be co-registered to high accuracy in order to
avoid spurious signals resulting from misalignment. Some of
the required preprocessing steps were discussed in Chapter~5. 
After having performed the necessary preprocessing, it is common
to examine various functions of the spectral bands involved
(differences, ratios or linear combinations) which in some way
bring the change information contained within them to the fore.
Large changes are often evident at a glance. However, other changes may
have occurred between the acquisition times and require more image processing to be clearly distinguished. 
Chapter 9 describes some commonly used  techniques for enhancing
change signals in bi-temporal satellite images and then focuses attention  on the  multivariate alteration
detection (MAD) algorithm  for visible/infrared imagery and on a sequential change statistic for polarimetric 
SAR data based on the complex Wishart distribution. The Chapter
concludes with an inverse application of change detection,
in which  unchanged pixels are used for automatic
relative radiometric normalization of multi-temporal imagery.
