--------------------------------------
MOVIE Software Usage
By: Kalpana Seshadrinathan 
Email: kalpana.seshadrinathan@ieee.org
--------------------------------------

MOVIE operates on reference and distorted videos provided in planar
YUV 4:2:0 format. 

Note that MOVIE scores tend to span a narrow range of values since the
model is non-linear and utilizes divisive normalization. We recommend
fitting MOVIE scores to subjective data using logistic fitting
functions (as described in the paper) for calibration. We have found
that multiplying MOVIE values by a factor of 100 aids in numerical
convergence of fitting algorithms when subjective scores lie in the
range of [0,100].

This software is best used in systems with RAM equal to or larger than
2GB (our testing environment used 2GB RAM systems). Execution speed
will suffer in systems with less RAM.

Command Line Arguments:
-----------------------

MOVIE requires the following command line arguments:

movie <Reference file> <Distorted file> <Reference file stem>
<Distorted file stem> <temp path> <output path> <Video width> <Video
height>

<Reference file> - reference video file in planar YUV420 format

<Distorted file> - distorted video file in planar YUV420 format

<Reference file stem> - a name for the reference video, used to prefix
filenames of temporary outputs 

<Distorted file stem> - a name for the distorted video, used to prefix
filenames of temporary outputs 

<temp path> - Path for storing intermediate results 

<output path> - Path for storing MOVIE outputs 

<Video width> - Width of input video in pixels 

<Video height> - Height of input video in pixels

Additionally, MOVIE accepts the following optional arguments:

<-f filtered image path> - Path that contains previously computed
results. Due to the computational complexity of MOVIE, this option is
provided to re-use results computed from the reference image (such as
Gabor filtered outputs and optical flow vectors), which is useful when
MOVIE is computed for multiple distorted videos obtained from the same
reference.

<-framestart start> - Frame to start MOVIE computation. Must be > 17.

<-framend end> - Last frame for MOVIE computation. 

<-frameint interval> - Interval between frames on which MOVIE is run,
which is recommended to be unaltered from the default value of 8.

<-remove> - delete some intermediate files created in <temp path> at
the end of the run. This command does not delete any reference video
outputs (which can be re-used using the "-f" option) and only deletes
distorted video outputs. It is recommended to use this option always
since MOVIE uses large amounts of disk space.

NOTE: non-minus arguments must be given in the specified order.

Suggested Usage:
----------------

./movie bs1_25fps.yuv bs2_25fps.yuv bs1_25fps bs2_25fps results/temp
results/outputs 768 432 -f results/temp -remove

This example shows how to run MOVIE on one of the videos in the LIVE
Video Quality Database available from
http://live.ece.utexas.edu/research/quality/live_video.html.

The reference and distorted filestems can be the names of the video
files (used without the extensions in this example). We strongly
recommend that the "-f" option be used whenever MOVIE is to be
computed for multiple distorted videos obtained from the same
reference. <filtered image path> for the "-f" option should be
identical to <temp path> in previous runs. We also recommend that the
"-remove" option be used if disk space is a concern, since MOVIE uses
large amounts of disk space to store temporary outputs.

Results:
--------

This program outputs a value for the MOVIE index in the terminal
window for the distorted video at the end of the run (which can take
several hours due to the complexity of the algorithm). This MOVIE index
for the distorted video is also stored in a text file "<distorted file
stem>_movie.txt" in <output path>.

This program also outputs two text files - <distorted file
stem>_smovie.txt, <distorted file stem>_tmovie.txt - in <output
path>. These files contain Spatial and Temporal MOVIE values
for each frame on which MOVIE is run respectively.

Additionally, this program outputs binary files containing spatial and
temporal MOVIE maps - "<distorted file stem>.smoviemap", "<distorted
file stem>.tmoviemap" - for each frame on which it is run in <output
path> (see paper for details). A MATLAB script ("read_movie_maps.m")
for reading in these maps and viewing them is also provided. The
program also outputs text files - <distorted file
stem>_smovie.frame#.txt, <distorted file stem>_tmovie.frame#.txt - in
<output path>. These files contains Spatial and Temporal MOVIE values
for frame "#" respectively. The MOVIE maps and the text files
containing Spatial and Temporal MOVIE values for frame "#" can be
accessed as soon as the frame computation finishes.
