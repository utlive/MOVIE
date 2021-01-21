/*------------------------------------------------------------------*/
/*  MOVIE Index, Version 1.0                                        */
/*  Copyright(c) 2009 Kalpana Seshadrinathan                        */ 
/*  All Rights Reserved.                                            */ 
/*------------------------------------------------------------------*/

/*------------------------------------------------------------------*/
/* Please cite the following paper in any published work if you use
   this software: 

   1. K. Seshadrinathan and A. C. Bovik, "Motion Tuned Spatio-temporal
   Quality Assessment of Natural Videos", vol. 19, no. 2, pp. 335-350,
   IEEE Transactions on Image Processing, Feb. 2010.

/* This is an implementation of the MOVIE index to compute the visual
   quality of a distorted video in the presence of a reference. Please
   refer to the above paper. */

/* The author was with the Laboratory for Image and Video Engineering,
   The University of Texas at Austin when this work was performed. She
   is currently with Intel Corporation, Chandler, AZ.               */
/*------------------------------------------------------------------*/

/*------------------------------------------------------------------*/
/* Permission to use, copy, or modify this software and its
   documentation for educational and research purposes only and
   without fee is hereby granted, provided that this copyright notice
   and the original authors' names appear on all copies and supporting
   documentation. This program shall not be used, rewritten, or
   adapted as the basis of a commercial software or hardware product
   without first obtaining permission of the authors. The authors make
   no representations about the suitability of this software for any
   purpose. It is provided "as is" without express or implied
   warranty.                                                         */
/*-------------------------------------------------------------------*/

/*-------------------------------------------------------------------*/
/* An implementation of optical flow estimation using the Fleet and
   Jepson algorithm was graciously provided by John Barron to the
   authors, which was extensively modified by Kalpana Seshadrinathan
   and used here. The authors are grateful to John Barron for
   providing "fleet.c" (by Travis Burkitt, November 29, 1988 --
   extensively modified by John Barron, 1991).                       */
/*-------------------------------------------------------------------*/

/*-------------------------------------------------------------------*/
/* Kindly report any suggestions or corrections to
   kalpana.seshadrinathan@ieee.org */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <complex>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics_double.h>

#define PI M_PI

#define TRUE 1
#define FALSE 0 

#define THRESHOLDED 0
#define NOT_THRESHOLDED 1
#define PRINT 0 /* Print debugging results */
#define PRINTFLOW 0
#define NFILTERS 35 /* number of filters per scale */

#define K_DC 1
#define K_SUM 100 
#define K_SP 0.1

#define SPMOVIE_WINDOW 7 /* Size of local window used for spatial MOVIE computation */
#define TMOVIE_WINDOW 7 /* Size of local window used for temporal MOVIE computation */

#define IS_VEL 1
#define UNDEFINED 100.0
#define WINDOW 2 /* number of pixels surrounding a given one that are used in the 2d velocity computation */
#define NUM_UNKNOWNS 6
#define TOL 0.000001

using namespace std;
typedef complex<double> dcmplx;

/*************************************************************/
/*                 Function declarations                     */
/*************************************************************/

double*** read_images(int* pic_size,string inputPath,string filename,int central_image,int offset);
void initFilterParameters(int num_scales, int* size, double* radius, double* sigma);
double*** alloc3ddoublearray(int nrows, int ncols, int nframes);
void dealloc3ddoublearray(double ***mat, int nrows, int ncols, int nframes);
dcmplx*** alloc3dcomplexarray(int nrows, int ncols, int nframes);
void dealloc3dcomplexarray(dcmplx ***mat, int nrows, int ncols, int nframes);
dcmplx** alloc2dcomplexarray(int,int);
void dealloc2dcomplexarray(dcmplx**,int,int);
double** alloc2ddoublearray(int,int);
void dealloc2ddoublearray(double**,int,int);
unsigned char** alloc2dchararray(int,int);
void dealloc2dchararray(unsigned char **,int,int);
unsigned char*** alloc3dchararray(int,int,int);
void dealloc3dchararray(unsigned char ***,int,int,int);
int read_yuv420_videos(string vidname,string vidbase,string tempPath,int* pic_size);
void read_image_files(double ***img,string inputPath,string filename,int* pic_size,int startfile,int endfile);
void read_full_velocities(double ***full_velocities,string flowPath,string flowname);
void filter_images(double ***ref_img,double*** dis_img,string tempPath,string outputPath,string refname,string disname,int* pic_size,int num_scales,int central_image);
void filter_images(double ***ref_img,double*** dis_img,string tempPath,string filtPath,string outputPath,string refname,string disname,int* pic_size,int num_scales,int central_image);
double** init_filters(double *radius,int num_scales);
void write_filtered_files(string filterPath,string filename,int *pic_size,int offset,dcmplx **filt_output);
void read_filtered_files(string filterPath, string filename,int *pic_size,int& offset,dcmplx **filt_output);
void outputFilterInfo(double* center);
void compute_movie(string tempPath,string outPath,string refname,string disname,int *pic_size,int num_scales,int central_image,double &smovie,double &tmovie);
void initialize_values(dcmplx **t, int nrows, int ncols, int offset, dcmplx val1, dcmplx val2);
void initialize_values(double **t, int nrows, int ncols, int offset, double val1, double val2);
void initialize_values(unsigned char **t, int nrows, int ncols,int offset, unsigned char val1, unsigned char val2);
void initialize_values(unsigned char ***t, int nrows, int ncols, int nframes,int offset, unsigned char val1, unsigned char val2);

double compute_dist_from_plane(double vx,double vy,double *filter_centerfreq);
void write_movie_maps(string filename,double **movie_maps,int *pic_size,int offset);
void write_movie_values(string filename,double *movie_values,int size);
void write_movie_values(string filename,double& movie_values);
void read_movie_maps(string fileout,double **movie_values,int *pic_size, int& offset);
void separable_convolve(dcmplx **gabor_output,double ***ref_img,dcmplx *filter_x,dcmplx *filter_y,dcmplx *filter_tt,int *pic_size,int filter_x_size,int filter_y_size,int filter_t_size);
void separable_convolve(double **gabor_output,double ***ref_img,double *filter_x,double *filter_y,double *filter_tt,int *pic_size,int filter_x_size,int filter_y_size,int filter_t_size);
void separable_convolve(double **gabor_output,double **ref_img,double *filter_x,double *filter_y,int *pic_size,int filter_x_size,int filter_y_size);
void conv_x(dcmplx **gabor_output,dcmplx **ref_img,dcmplx *filter_x,int *pic_size,int filter_x_size);
void conv_y(dcmplx **gabor_output,dcmplx **ref_img,dcmplx *filter_x,int *pic_size,int filter_x_size);
void conv_tt(dcmplx **gabor_output,double ***ref_img,dcmplx *filter_x,int *pic_size,int filter_x_size);
void conv_x(double **gabor_output,double **ref_img,double *filter_x,int *pic_size,int filter_x_size);
void conv_y(double **gabor_output,double **ref_img,double *filter_x,int *pic_size,int filter_x_size);
void conv_tt(double **gabor_output,double ***ref_img,double *filter_x,int *pic_size,int filter_x_size);
void create_gabor_1d(dcmplx *filter,double center_freq,double sigma,int len);
void create_gauss_1d(double *filter,double sigma,int len);
void subImages(dcmplx** img1, double** img2, int* pic_size, double alpha);
void subImages(double** img1, double** img2, int* pic_size);
double calc_maxamp(dcmplx**,int*, double**);
double calc_maxamp(double **filt_output,int *pic_size, double** max_response);
double read_max_value(string filterPath,string s);
void write_max_value(string filterPath,string s,double maxamp);
void write_velocity_response(string filename,double **vel_response,int *vel_size, int offset);
void read_velocity_response(string filename,double **vel_response,int *vel_size, int& offset);


void thresh_and_compute(string,string,string,int*,int,int,double,double);
void initialize_thresholds(unsigned char***,int,int,int);
void initialize_full_velocities(double***,int,int,int);
void threshold(string,string,int*,int,unsigned char***,dcmplx***,double,double,int);
void compute_derivatives(double ***img,string filename,string tempPath,int *pic_size,int num_scales,int central_image);
void compute_derivatives(double ***img,string filename,string tempPath,string filtPath,int *pic_size,int num_scales,int central_image);
void produce_thresh_report(long int*,long int*,long int*,int,long int);
void compute_full_velocities(double ****,double ***,dcmplx***,unsigned char***,int*,int);
double cal_velocity(int,double**,double*,double[3][3][2],double*);
void output_full_velocities(const string,const string,int*,int,double***);
void create_derivative_kernel(dcmplx *filter, double center_freq,double sigma,int len);

int test_full_velocities(int *pic_size,string flowPath,string flowname);
void design_weights(double ***weights,double ***full_velocities,string outPath,string refname,int *pic_size,int num_scales,int centralFramenum);
void check_spatial(string tempPath,string outPath,string refname,string disname,int framenum,int num_scales,int *is_spatial,double ***movie_spatial);
void check_temporal(string tempPath,string refname,string disname, int framenum, int& is_ref_vel, int& is_dis_vel,double **ref_vel_num,double **ref_vel_den,double **dis_vel_num,double **dis_vel_den);

/*************************************************************/
/*                 Main                                      */
/*************************************************************/

int main(int argc, char* argv[]){

  int remove = 0;
  int pic_size[3]; //Size of the video sequence
  double ***ref_img,***dis_img; //double array that contains the video sequence
  int* size; //array that contains the dimensions of filters at each scale in space
  double* radius; //array that contains the radius of filters at each scale
  double* sigma; //array that contains sigma of filters in space at each scale
  int offset; //Number of files about central image that we want to read
  int frame_start = 0; 
  int frame_end = 0;
  int frame_int = 8; //interval between frames on which MOVIE is run
  int num_scales = 3; //number of scales of filters

  double tau,percent_maxamp; //parameters of Fleet and Jepson algorithm
  tau = 1.25;
  percent_maxamp = 0.05;

  int minArgs = 9;

  if(argc < minArgs)
    {
      cout << "Usage: "<<argv[0]<<" <Reference file> <Distorted file> <Reference file stem> <Distorted file stem>\n";
      cout << "<temp path> <output path> <Video width> <Video height> \n";
      cout << "<Reference file> - reference file in planar YUV420 format\n";
      cout << "<Distorted file> - distorted file in planar YUV420 format\n";
      cout << "<Reference file stem> - the reference name stem\n";
      cout << "<Distorted file stem> - the distorted name stem \n";
      cout << "<temp path> - Where intermediate results are stored \n";
      cout << "<output path> - where the output goes\n";
      cout << "<Video width> - Width of input video\n";
      cout << "<Video height> - Height of input video\n";
      cout << "<-f filtered image path> - Path for filtered images if they are already computed \n";
      cout << "<-framestart interval> - Frame to start MOVIE computation, must be > 17\n";
      cout << "<-framend interval> - Last frame for MOVIE computation\n";
      cout << "<-frameint interval> - Interval between frames on which MOVIE is run, default is 8 \n";
      cout << "<-remove> - delete all files in <temp path>";
      cout << "NOTE: non-minus arguments must be given in the specified order\n";
      cout << argc << "  arguments specified\n";
      exit(1);
    }

  if(SPMOVIE_WINDOW%2!=1){
    cout << "Window size specified in SPMOVIE_WINDOW has to be odd \n";
    cout << "Exiting... \n";
    exit(-1);
  }

  /* Compute size of the input data from sigma */

  const string refvid(argv[1]);
  const string disvid(argv[2]);
  const string refname(argv[3]);
  const string disname(argv[4]);
  const string tempPath(argv[5]);
  const string outputPath(argv[6]);
  pic_size[1] = atoi(argv[7]);
  pic_size[0] = atoi(argv[8]);

  string filtPath;
  int isFiltered = FALSE; //No path for filtered images provided
  int isFlow = FALSE;

  int argCount = minArgs;

  while(argCount < argc){
    if(strcmp(argv[argCount],"-f") == 0){
      filtPath.assign(argv[argCount+1]);
      isFiltered = TRUE;
      argCount+=2;
    }
    else if(strcmp(argv[argCount],"-remove") == 0){
      remove = 1;
      argCount+=1;
    }
    else if(strcmp(argv[argCount],"-framestart") == 0){
      frame_start = atoi(argv[argCount+1]);
      if(frame_start<17){
	frame_start = 17;
      }
      argCount+=2;
    }
    else if(strcmp(argv[argCount],"-framend") == 0){
     
      frame_end = atoi(argv[argCount+1]);
      if(frame_end < frame_start){
	cout << "End frame is less than start frame; won't do anything!\n";
      }
      argCount+=2;
    }
    else if(strcmp(argv[argCount],"-frameint") == 0){
      frame_int = atoi(argv[argCount+1]);
      argCount+=2;
    }
    else{
      cout << "Unknown command line argument specified. Exiting...\n";
      exit(-1);
    }
  }

  fflush(stdout);

  /* Allocate the center frequencies and the bandwidths of the filters */

  size = new int[num_scales];
  radius = new double[num_scales];
  sigma = new double[num_scales];

  /* Initialize the filter parameters */
  initFilterParameters(num_scales,size,radius,sigma);
  cout << "\nInitialized filter parameters \n";
  
  offset = size[num_scales-1]/2; //Number of frames to be read in depends on size of filters at coarsest scale

  /* Read in the reference video */
  int ref_frames = read_yuv420_videos(refvid,refname,tempPath,pic_size);
  cout << "Reference video has been read in, " << ref_frames << " frames\n";

  /* Read in the distorted video */
  int dis_frames = read_yuv420_videos(disvid,disname,tempPath,pic_size);
  cout << "Distorted video has been read in, " << dis_frames << " frames\n";

  int min_frames = gsl_min(ref_frames,dis_frames);
  assert(min_frames >= 33);

  ostringstream framestr;

  if(frame_start==0){
    frame_start = 17;
  }
  if(frame_end==0){
    frame_end = min_frames-16;
  }

  int movie_frames = (frame_end-frame_start)/frame_int + 1;
  assert(movie_frames>0);

  double *smovie_perframe = new double[movie_frames];
  double *tmovie_perframe = new double[movie_frames];
  int count = 0;

  for (int central_image = frame_start; central_image<=frame_end; central_image+=frame_int){

    assert(count < movie_frames);

    cout << "\n\n---------------------------------------------\n";
    cout << "Computing MOVIE for frame " << central_image << "\n";
    cout << "---------------------------------------------\n\n";

    /* Read in the reference images */
    ref_img = read_images(pic_size,tempPath,refname,central_image,offset);
    cout << "Reference image files have been read in... \n";

    /* Read in the distorted images */
    dis_img = read_images(pic_size,tempPath,disname,central_image,offset);
    cout << "Distorted image files have been read in... \n";

    /* Compute filtered results for reference and distorted images */
    if(isFiltered){
      /* Filter the reference image */
      filter_images(ref_img,dis_img,tempPath,filtPath,outputPath,refname,disname,pic_size,num_scales,central_image);
    }

    else{
      /* Filter the reference image */
      filter_images(ref_img,dis_img,tempPath,outputPath,refname,disname,pic_size,num_scales,central_image);
    }

    /* Compute derivatives and flow field according to Fleet and Jepson algorithm*/

    /* Check if the filtered path provided has the file with the right name and right number of bytes */
    if(isFiltered){
      ostringstream framenum;
      framenum << central_image;
      string flowname = refname + ".frame" + framenum.str() + ".finalflow";
      if(!test_full_velocities(pic_size,filtPath,flowname)){
	isFlow=FALSE;
      }
      else{
	isFlow = TRUE;
	cout << "\nFlow field found in possible path \n";
      }
    }

    /* Perform the flow computation if no/inaccurate flow information is provided*/

    if(!isFlow){
      
      if(isFiltered){
	compute_derivatives(ref_img,refname,tempPath,filtPath,pic_size,num_scales,central_image);
      }
      else{
	compute_derivatives(ref_img,refname,tempPath,pic_size,num_scales,central_image);
      }

      dealloc3ddoublearray(ref_img,pic_size[0],pic_size[1],pic_size[2]);
      dealloc3ddoublearray(dis_img,pic_size[0],pic_size[1],pic_size[2]);
      cout << "\nComputing flow... \n";

      /* Threshold filtered results and compute velocities */
      thresh_and_compute(refname,tempPath,outputPath,pic_size,central_image,num_scales,tau,percent_maxamp);
    }
    else{

      dealloc3ddoublearray(ref_img,pic_size[0],pic_size[1],pic_size[2]);
      dealloc3ddoublearray(dis_img,pic_size[0],pic_size[1],pic_size[2]);
    }

    /* Compute MOVIE values for the frame at each pixel location */

    cout << "\nComputing MOVIE indices... \n";

    if(isFiltered){  
      compute_movie(filtPath,outputPath,refname,disname,pic_size,num_scales,central_image,smovie_perframe[count],tmovie_perframe[count]);
      cout << "\nFrame: " << central_image << " - Spatial MOVIE = " << smovie_perframe[count] << ", Temporal MOVIE = " << tmovie_perframe[count] << endl;
      count++;
    }
    else{
      compute_movie(tempPath,outputPath,refname,disname,pic_size,num_scales,central_image,smovie_perframe[count],tmovie_perframe[count]);
      cout << "\nFrame: " << central_image << " - Spatial MOVIE = " << smovie_perframe[count] << ", Temporal MOVIE = " << tmovie_perframe[count] << endl;
      count++;
    }

    framestr.str("");
    framestr << central_image;
    write_movie_values(outputPath+"/"+disname+"_smovie.frame"+framestr.str()+".txt",smovie_perframe[count-1]);
    write_movie_values(outputPath+"/"+disname+"_tmovie.frame"+framestr.str()+".txt",tmovie_perframe[count-1]);

    if(remove){

#ifndef unix

cout << "Deleting intermediate files for distorted video for frame " << framestr.str() << "...\n";
      string s = "del " + tempPath + "\\" + disname + "*" + "frame" + framestr.str() + "*";
      cout << "Executing command: " <<  s << endl;
      system(s.c_str());

      s = "del " + tempPath + "\\" + refname + "*" + "frame" + framestr.str() + ".gabx*";
      system(s.c_str());
      s = "del " + tempPath + "\\" + refname + "*" + "frame" + framestr.str() + ".gaby*";
      system(s.c_str());
      s = "del " + tempPath + "\\" + refname + "*" + "frame" + framestr.str() + ".gabt*";
      system(s.c_str());

#else

      cout << "Deleting intermediate files for distorted video for frame " << framestr.str() << "...\n";
      string s = "rm " + tempPath + "/" + disname + "*" + "frame" + framestr.str() + "*";
      cout << "Executing command: " <<  s << endl;
      system(s.c_str());

      s = "rm " + tempPath + "/" + refname + "*" + "frame" + framestr.str() + ".gabx*";
      system(s.c_str());
      s = "rm " + tempPath + "/" + refname + "*" + "frame" + framestr.str() + ".gaby*";
      system(s.c_str());
      s = "rm " + tempPath + "/" + refname + "*" + "frame" + framestr.str() + ".gabt*";
      system(s.c_str());

#endif
    }
  }

  assert(count==movie_frames);

  write_movie_values(outputPath+"/"+disname+"_smovie.txt",smovie_perframe,movie_frames);
  write_movie_values(outputPath+"/"+disname+"_tmovie.txt",tmovie_perframe,movie_frames);

  double sp_movie = gsl_stats_mean(smovie_perframe,1,movie_frames);
  double t_movie = gsl_stats_mean(tmovie_perframe,1,movie_frames);

  double movie = sp_movie*sqrt(t_movie);

  write_movie_values(outputPath+"/"+disname+"_movie.txt",movie);

  if(remove){
#ifndef unix
cout << "Deleting intermediate files for distorted video...\n";
    string s = "del " + tempPath + "\\" + disname + "*";
    cout << "Executing command: " <<  s << endl;
    system(s.c_str());
#else
    cout << "Deleting intermediate files for distorted video...\n";
    string s = "rm " + tempPath + "/" + disname + "*";
    cout << "Executing command: " <<  s << endl;
    system(s.c_str());
#endif
  }

  cout << "\n\n\n------------------------------------------------\n\n";
  cout << "MOVIE computation COMPLETED!!!\n\n";
  cout << "Spatial MOVIE value = " << sp_movie << endl;
  cout << "Temporal MOVIE value = " << t_movie << endl;
  cout << "MOVIE value = " << movie << endl;
  cout << "\n\n\n------------------------------------------------\n\n";


  delete [] smovie_perframe;
  delete [] tmovie_perframe;
  delete [] size;
  delete [] radius;
  delete [] sigma;
  
}

/*********************************************************************************************************************/
/*    Tests if file provided has the right number of bytes for a flow field                                          */
/*********************************************************************************************************************/

int test_full_velocities(int *pic_size,string flowPath,string flowname){

  string s = flowPath+"/"+flowname;

  ifstream in;
  in.open(s.c_str(),ios::in | ios::binary);
  if(in.fail()){
    cout << "Unable to open file: " << s << endl;
    return 0;
  }

  int nrows,ncols,off;

  /* First, read in the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  in.read(reinterpret_cast<char*>(&nrows), sizeof(int));
  in.read(reinterpret_cast<char*>(&ncols),sizeof(int));
  in.read(reinterpret_cast<char*>(&off),sizeof(int));

  in.seekg(0,ios::end);
  int filesize = in.tellg();

  int expectedFileLength = 3*sizeof(int)+2*(ncols-2*off)*(nrows-2*off)*sizeof(double);
  if(filesize!=expectedFileLength){
    cout << "File size read: " << filesize << ", File size expected: " << expectedFileLength << endl;
    in.close();
    return 0;
  } 
  else{
    in.close();
    return 1;
  }
}

/********************************************************************/
/* Read in the images ****************************/
/********************************************************************/

double*** read_images(int* pic_size,string inputPath,string filename,int central_image,int offset)
{

  int startfile,endfile;
  double ***img;
  static int test = 0;

  pic_size[2] = 2*offset+1;

  /* Allocate memory for an array to read in the image files */
  img = alloc3ddoublearray(pic_size[0],pic_size[1],pic_size[2]);

  startfile = central_image-offset;
  endfile = central_image+offset;

  /* Read in the image files */
  read_image_files(img,inputPath,filename,pic_size,startfile,endfile);

  if(PRINT){
    cout << "STARTFILE: " << startfile <<"\t ENDFILE: " << endfile << endl;  
    if(test == 0){
      cout << "Image values for the first frame \n";
      for(int i=0; i<pic_size[0]; i++){
	for(int j=0; j<pic_size[1];j++){
	  cout << img[0][i][j] << "\t";
	}
	cout << "\n\n";
      }
      test = 1;
    }
  }

  return img;
}

/**********************************************************************/
/**      Read in the image video file - planar YUV format             */
/**********************************************************************/

int read_yuv420_videos(string vidname,string vidbase,string tempPath,int* pic_size)
{

  ifstream in;
  ofstream out;
  unsigned char *buf = new unsigned char[pic_size[1]*pic_size[0]];

  string filein = vidname; 
  in.open(filein.c_str(),ios::in | ios::binary);
  assert(in);

  int count = 0;
  ostringstream countstr;

  while(!in.eof()){

    in.read(reinterpret_cast<char *>(buf),pic_size[1]*pic_size[0]);    
    if(!in){
      in.close();
      delete [] buf;	
      return count;
    }

    count = count + 1;
    countstr.str("");
    countstr << count;
    string frameout = tempPath + "/" +  vidbase + "." + countstr.str();
    out.open(frameout.c_str(),ios::out | ios::binary);
    assert(out);
    out.write(reinterpret_cast<char *>(buf),pic_size[0]*pic_size[1]);
    out.close();

    in.seekg(count*1.5*pic_size[0]*pic_size[1]);

  }

  in.close();
  delete [] buf;
  return count;
  
}

/**********************************************************************/
/************** Read in the image files into the 3d array  ************/
/**********************************************************************/

void read_image_files(double ***img,string inputPath,string filename,int* pic_size,int startfile,int endfile)
{

  ifstream in;
  unsigned char *buf = new unsigned char[pic_size[1]];

  for(int i=startfile; i<=endfile; i++){

    /* Name of file is inputPath/filename.i */

    /* convert integer to a number */
    ostringstream filenum;
    filenum << i;

    string filein = inputPath + "/" + filename + "." + filenum.str(); 
    in.open(filein.c_str(),ios::in | ios::binary);

    assert(in);

    /* Image data is stored column wise */

    for(int k=0; k<pic_size[0]; k++){ //row index
      in.read(reinterpret_cast<char *>(buf),pic_size[1]);
      for(int j=0; j<pic_size[1]; j++){ //column index
	img[i-startfile][k][j] = static_cast<double>(buf[j]);
      }
    }

    in.close();
  }

  delete [] buf;

}

/*********************************************************************/
/* Initialize filter parameters **************************************/
/*********************************************************************/

void initFilterParameters(int num_scales, int* size, double* radius, double* sigma)
{

  const double f0 = 0.7*PI;
  const double beta = 0.5; 

  /* Determine center frequencies of the filters and allocate the size of the filters at each scale */

  for (int i=0; i<num_scales; i++){

    /* Initialize center frequency in radians */
    radius[i] = f0/pow(pow(2,beta),i);

    /* Initialize the std. devn of the gabor in space */
    double sigma_freq;
    sigma_freq = (radius[i]*(pow(2,beta)-1))/(pow(2,beta)+1);
    sigma[i] = 1.0/sigma_freq;

    /* Initialize the size of the filters in space */
    size[i] = static_cast<int>(6*sigma[i]+1);
    if(size[i]%2 == 0){ 
      size[i] += 1;
    }
  }
}

/**********************************************************************/
/* Initializes the center frequencies of the gabor filters ************/
/**********************************************************************/

double** init_filters(double *radius,int num_scales){

  double speed1,speed2;
  double **tuning_info, **filter_centerfreq;
  double theta,rho,speed,k1,k2,k3;

  speed2 = sqrt(3.0);
  speed1 = 1.0/speed2;

  const double two_pi = 2*PI;

  tuning_info = new double*[NFILTERS];
  for (int i=0; i<NFILTERS; i++){
    tuning_info[i] = new double[2];
  }

  tuning_info[0][0] = 0.0; tuning_info[0][1] = 0.0;
  tuning_info[1][0] = 20.0; tuning_info[1][1] = 0.0;
  tuning_info[2][0] = 40.0; tuning_info[2][1] = 0.0;
  tuning_info[3][0] = 60.0; tuning_info[3][1] = 0.0;
  tuning_info[4][0] = 80.0; tuning_info[4][1] = 0.0;
  tuning_info[5][0] = 100.0; tuning_info[5][1] = 0.0;
  tuning_info[6][0] = 120.0; tuning_info[6][1] = 0.0;
  tuning_info[7][0] = 140.0; tuning_info[7][1] = 0.0;
  tuning_info[8][0] = 160.0; tuning_info[8][1] = 0.0;

  tuning_info[9][0] = 0.0; tuning_info[9][1] = speed1;
  tuning_info[10][0] = 22.0; tuning_info[10][1] = speed1;
  tuning_info[11][0] = 44.0; tuning_info[11][1] = speed1;
  tuning_info[12][0] = 66.0; tuning_info[12][1] = speed1;
  tuning_info[13][0] = 88.0; tuning_info[13][1] = speed1;
  tuning_info[14][0] = 110.0; tuning_info[14][1] = speed1;
  tuning_info[15][0] = 132.0; tuning_info[15][1] = speed1;
  tuning_info[16][0] = 154.0; tuning_info[16][1] = speed1;
  tuning_info[17][0] = 176.0; tuning_info[17][1] = speed1;
  tuning_info[18][0] = 198.0; tuning_info[18][1] = speed1;
  tuning_info[19][0] = 220.0; tuning_info[19][1] = speed1;
  tuning_info[20][0] = 242.0; tuning_info[20][1] = speed1;
  tuning_info[21][0] = 264.0; tuning_info[21][1] = speed1;
  tuning_info[22][0] = 286.0; tuning_info[22][1] = speed1;
  tuning_info[23][0] = 308.0; tuning_info[23][1] = speed1;
  tuning_info[24][0] = 330.0; tuning_info[24][1] = speed1;
  tuning_info[25][0] = 352.0; tuning_info[25][1] = speed1;

  tuning_info[26][0] = 0.0; tuning_info[26][1] = speed2;
  tuning_info[27][0] = 40.0; tuning_info[27][1] = speed2;
  tuning_info[28][0] = 80.0; tuning_info[28][1] = speed2;
  tuning_info[29][0] = 120.0; tuning_info[29][1] = speed2;
  tuning_info[30][0] = 160.0; tuning_info[30][1] = speed2;
  tuning_info[31][0] = 200.0; tuning_info[31][1] = speed2;
  tuning_info[32][0] = 240.0; tuning_info[32][1] = speed2;
  tuning_info[33][0] = 280.0; tuning_info[33][1] = speed2;
  tuning_info[34][0] = 320.0; tuning_info[34][1] = speed2;

  /* allocate memory to compute the center frequencies of all the filters, includes a DC */

  filter_centerfreq = new double*[num_scales*NFILTERS+1];
  for(int i=0; i<num_scales*NFILTERS+1; i++){
    filter_centerfreq[i] = new double[3];
  }

  for(int i=0; i<num_scales; i++){
    for(int j=0; j<NFILTERS; j++){

      theta = tuning_info[j][0];
      speed = tuning_info[j][1];

      rho = radius[i]/sqrt(pow(speed,2)+1);
      k3 = rho*speed; 

      k1 = rho*cos(theta*two_pi/360.0);
      k2 = rho*sin(theta*two_pi/360.0);

      filter_centerfreq[i*NFILTERS+j][0] = k1;
      filter_centerfreq[i*NFILTERS+j][1] = k2;
      filter_centerfreq[i*NFILTERS+j][2] = k3;

    }
  }

  filter_centerfreq[num_scales*NFILTERS][0] = 0;
  filter_centerfreq[num_scales*NFILTERS][1] = 0;
  filter_centerfreq[num_scales*NFILTERS][2] = 0;

  for(int i=0; i<NFILTERS; i++){
    delete [] tuning_info[i];
  }

  delete [] tuning_info;

  return filter_centerfreq;

}

/*****************************************************************/
/* Outputs the filter frequency information **********************/
/*****************************************************************/

void outputFilterInfo(double* center)
{
  cout << "Center frequency: ";
  cout.width(14);
  cout << center[0] << " ";
  cout.width(14);
  cout << center[1] << " ";
  cout.width(14);
  cout << center[2] << endl;
}

/*********************************************************************/
/*********** Allocate a 3d double array of given size ****************/
/********************************************************************/

double*** alloc3ddoublearray(int nrows, int ncols, int nframes)
{

  double*** mat;
  mat = new double**[nframes];

  if(mat==0)
    cout << "Unable to allocate memory for frames of 3d matrix \n";

  for(int i=0; i<nframes; i++){
    mat[i] = new double*[nrows];
    if(mat[i]==0)
      cout << "Unable to allocate memory for rows of 3d matrix \n";
    for(int j=0; j<nrows; j++){
      mat[i][j] = new double[ncols];
      if(mat[i][j]==0)
	cout << "Unable to allocate memory for columns of 3d matrix \n";
    }
  }

  return mat;
}

/*******************************************************************/
/***************** Deallocate a 3d double array *********************/
/*******************************************************************/

void dealloc3ddoublearray(double ***mat, int nrows, int ncols, int nframes)
{

  for(int i=0; i<nframes; i++){
    for(int j=0; j<nrows; j++){
      delete [] mat[i][j];
    }
    delete [] mat[i];
  }

  delete [] mat;
}

/*********************************************************************/
/*********** Allocate a 3d complex array of given size ****************/
/********************************************************************/

dcmplx*** alloc3dcomplexarray(int nrows, int ncols, int nframes)
{

  dcmplx*** mat;
  mat = new dcmplx**[nframes];

  if(mat==0)
    cout << "Unable to allocate memory for frames of 3d matrix \n";

  for(int i=0; i<nframes; i++){
    mat[i] = new dcmplx*[nrows];
    if(mat[i]==0)
      cout << "Unable to allocate memory for rows of 3d matrix \n";
    for(int j=0; j<nrows; j++){
      mat[i][j] = new dcmplx[ncols];
      if(mat[i][j]==0)
	cout << "Unable to allocate memory for columns of 3d matrix \n";
    }
  }
  return mat;
}

/*******************************************************************/
/***************** Deallocate a 3d complex array *********************/
/*******************************************************************/

void dealloc3dcomplexarray(dcmplx ***mat, int nrows, int ncols, int nframes)
{

  for(int i=0; i<nframes; i++){
    for(int j=0; j<nrows; j++){
      delete [] mat[i][j];
    }
    delete [] mat[i];
  }

  delete [] mat;
}

/*********************************************************************/
/*********** Allocate a 2d double array of given size ****************/
/********************************************************************/

double** alloc2ddoublearray(int nrows, int ncols)
{

  double** mat;
  mat = new double*[nrows];

  if(mat==0)
    cout << "Unable to allocate memory for rows of 2d matrix \n";

  for(int i=0; i<nrows; i++){
    mat[i] = new double[ncols];
    if(mat[i]==0)
      cout << "Unable to allocate memory for columns of 2d matrix \n";
  }
  return mat;
}

/*******************************************************************/
/***************** Deallocate a 2d double array *********************/
/*******************************************************************/

void dealloc2ddoublearray(double **mat, int nrows, int ncols)
{

  for(int i=0; i<nrows; i++){
    delete [] mat[i];
  }

  delete [] mat;
}

/*********************************************************************/
/*********** Allocate a 2d complex array of given size ****************/
/********************************************************************/

dcmplx** alloc2dcomplexarray(int nrows, int ncols)
{

  dcmplx** mat;
  mat = new dcmplx*[nrows];

  if(mat==0)
    cout << "Unable to allocate memory for rows of 2d matrix \n";

  for(int i=0; i<nrows; i++){
    mat[i] = new dcmplx[ncols];
    if(mat[i]==0)
      cout << "Unable to allocate memory for columns of 2d matrix \n";
  }
  return mat;
}

/*******************************************************************/
/***************** Deallocate a 2d complex array *********************/
/*******************************************************************/

void dealloc2dcomplexarray(dcmplx **mat, int nrows, int ncols)
{

  for(int i=0; i<nrows; i++){
    delete [] mat[i];
  }

  delete [] mat;
}

/*********************************************************************/
/*********** Allocate a 3d character array of given size ****************/
/********************************************************************/

unsigned char*** alloc3dchararray(int nrows, int ncols, int nframes)
{

  unsigned char*** mat;
  mat = new unsigned char**[nframes];

  if(mat==0)
    cout << "Unable to allocate memory for frames of 3d matrix \n";

  for(int i=0; i<nframes; i++){
    mat[i] = new unsigned char*[nrows];
    if(mat[i]==0)
      cout << "Unable to allocate memory for rows of 3d matrix \n";
    for(int j=0; j<nrows; j++){
      mat[i][j] = new unsigned char[ncols];
      if(mat[i][j]==0)
	cout << "Unable to allocate memory for columns of 3d matrix \n";
    }
  }

  return mat;

}

/*******************************************************************/
/***************** Deallocate a 3d character array *********************/
/*******************************************************************/

void dealloc3dchararray(unsigned char ***mat, int nrows, int ncols, int nframes)
{

  for(int i=0; i<nframes; i++){
    for(int j=0; j<nrows; j++){
      delete [] mat[i][j];
    }
    delete [] mat[i];
  }

  delete [] mat;
}

/*********************************************************************/
/*********** Allocate a 2d character array of given size ****************/
/********************************************************************/

unsigned char** alloc2dchararray(int nrows, int ncols)
{

  unsigned char** mat;
  mat = new unsigned char*[nrows];

  if(mat==0)
    cout << "Unable to allocate memory for rows of 2d matrix \n";

  for(int i=0; i<nrows; i++){
    mat[i] = new unsigned char[ncols];
    if(mat[i]==0)
      cout << "Unable to allocate memory for columns of 3d matrix \n";
  }

  return mat;

}

/*******************************************************************/
/***************** Deallocate a 2d character array *********************/
/*******************************************************************/

void dealloc2dchararray(unsigned char **mat, int nrows, int ncols)
{

  for(int i=0; i<nrows; i++){
    delete [] mat[i];
  }

  delete [] mat;
}

/*******************************************************************/
/*                Read in the flow field                           */
/*******************************************************************/

void read_full_velocities(double ***full_velocities,string flowPath,string flowname){

  string s = flowPath+"/"+flowname;

  ifstream in;
  in.open(s.c_str(),ios::in | ios::binary);
  assert(in);

  int nrows,ncols,off;

  /* First, read in the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  in.read(reinterpret_cast<char*>(&nrows), sizeof(int));
  in.read(reinterpret_cast<char*>(&ncols),sizeof(int));
  in.read(reinterpret_cast<char*>(&off),sizeof(int));

  /* Read in the flow values - u field followed by the v field*/

  for(int i=off; i<nrows-off; i++){
    in.read(reinterpret_cast<char*>(&full_velocities[0][i][off]),sizeof(double)*(ncols-2*off));
  }

  for(int i=off; i<nrows-off; i++){
    in.read(reinterpret_cast<char*>(&full_velocities[1][i][off]),sizeof(double)*(ncols-2*off));
  }

  in.close();
}

/*********************************************************************/
/* Filters the images with the set of Gabor filters ******************/
/*********************************************************************/

void filter_images(double ***ref_img,double ***dis_img,string tempPath,string outputPath,string refname,string disname,int* pic_size,int num_scales,int central_image)
{

  dcmplx *gabor_filter_x,*gabor_filter_y,*gabor_filter_tt;
  double *gauss_filter_x,*gauss_filter_y,*gauss_filter_tt;
  dcmplx **gabor_output;
  double **gauss_output;
  double **max_response;
  double **filter_centerfreq;

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];

  double maxamp = 0;

  /* Compute the center frequencies of the Gabor filters */
  initFilterParameters(num_scales,size,radius,sigma);
  filter_centerfreq = init_filters(radius,num_scales);

  /* Set DC parameters */

  double sigma_dc_freq = radius[num_scales-1]-1/sigma[num_scales-1];
  double sigma_dc = 1.0/sigma_dc_freq;
  int size_dc = static_cast<int>(6*sigma_dc + 1);
  if(size_dc%2 == 0){ 
    size_dc += 1;
  }

  const double dc_factor = 0.001;  
  int len,offset;
  int filter_size[3]; //[nrows,ncols,nframes]

  /* Declare variables to hold the output */

  gabor_output = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  gauss_output = alloc2ddoublearray(pic_size[0],pic_size[1]);
  max_response = alloc2ddoublearray(pic_size[0],pic_size[1]);
  initialize_values(max_response,pic_size[0],pic_size[1],0,0,0);

  ostringstream centralFramenum;
  centralFramenum << central_image;

  len = size_dc;
  offset = size_dc/2;
  filter_size[0] = filter_size[1] = filter_size[2] = len;
  cout << "Convolving video with DC filter:" << endl;
  
  gauss_filter_x = new double[len];
  gauss_filter_y = new double[len];
  gauss_filter_tt = new double[len];

  string refdc = refname + ".frame" + centralFramenum.str()+".dc";
  string disdc = disname + ".frame" + centralFramenum.str()+".dc";

  create_gauss_1d(gauss_filter_x,sigma_dc,len);
  create_gauss_1d(gauss_filter_y,sigma_dc,len);
  create_gauss_1d(gauss_filter_tt,sigma_dc,len);

  separable_convolve(gauss_output,ref_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);
  write_velocity_response(tempPath+"/"+refdc,gauss_output,pic_size,offset);

  separable_convolve(gauss_output,dis_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);
  write_velocity_response(tempPath+"/"+disdc,gauss_output,pic_size,offset);

  delete [] gauss_filter_x;
  delete [] gauss_filter_y;
  delete [] gauss_filter_tt;

  cout << "\n\nConvolving video with Gabor filters:";
  for(int i=0; i<num_scales; i++){

    cout << "\nScale " << i << ":\n";
    ostringstream scalenum;
    scalenum << i;

    len = size[i];
    offset = size[i]/2;
    filter_size[0] = filter_size[1] = filter_size[2] = len;

    /* Declare variables to hold the filter */
    gabor_filter_x = new dcmplx[len];
    gabor_filter_y = new dcmplx[len];
    gabor_filter_tt = new dcmplx[len];

    gauss_filter_x = new double[len];
    gauss_filter_y = new double[len];
    gauss_filter_tt = new double[len];

    for(int j=0; j<NFILTERS; j++){

      cout << i*NFILTERS+j << "\t";
      fflush(stdout);

      int filtnum = i*NFILTERS+j;

      for(int frameind=central_image;frameind <= central_image;frameind++){

	ostringstream framenum;
	framenum << frameind;

	/* Convert integer to string */
	ostringstream filenum;
	filenum << filtnum;
	string refbase = refname + ".scale" + scalenum.str() + ".frame" + framenum.str()+".gab"+filenum.str();
	string disbase = disname + ".scale" + scalenum.str() + ".frame" + framenum.str()+".gab"+filenum.str();

	/* Create the Gabor filter */
	create_gabor_1d(gabor_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	create_gabor_1d(gabor_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	create_gabor_1d(gabor_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);

	create_gauss_1d(gauss_filter_x,sigma[i],len);
	create_gauss_1d(gauss_filter_y,sigma[i],len);
	create_gauss_1d(gauss_filter_tt,sigma[i],len);

	separable_convolve(gabor_output,ref_img,gabor_filter_x,gabor_filter_y,gabor_filter_tt,pic_size,len,len,len);
	separable_convolve(gauss_output,ref_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);

	/* Subtract 0.001 of the gaussian from the cosine filter to remove dc component */
	subImages(gabor_output,gauss_output,pic_size,dc_factor);

	/* Write the outputs to a file */
	write_filtered_files(tempPath,refbase,pic_size,offset,gabor_output);

	double curr_max = calc_maxamp(gabor_output,pic_size,max_response);
	if(maxamp < curr_max){
	  maxamp = curr_max;
	}

	separable_convolve(gabor_output,dis_img,gabor_filter_x,gabor_filter_y,gabor_filter_tt,pic_size,len,len,len);
	separable_convolve(gauss_output,dis_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);

	/* Subtract 0.001 of the gaussian from the cosine filter to remove dc component */
	subImages(gabor_output,gauss_output,pic_size,dc_factor);

	/* Write the outputs to a file */
	write_filtered_files(tempPath,disbase,pic_size,offset,gabor_output);

      }
    }

    /* Deallocate all variables */

    delete [] gabor_filter_x;
    delete [] gabor_filter_y;
    delete [] gabor_filter_tt;
    delete [] gauss_filter_x;
    delete [] gauss_filter_y;
    delete [] gauss_filter_tt;

  }

  dealloc2dcomplexarray(gabor_output,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(gauss_output,pic_size[0],pic_size[1]);

  /* Deallocate the filter_centerfreq variable */

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }

  delete [] filter_centerfreq; 

  /* Write the max value to a file */

  string s = refname + ".frame" + centralFramenum.str() + ".max";  
  write_max_value(tempPath,s,maxamp);

  offset = size[num_scales-1]/2;
  string refmax = refname + ".frame" + centralFramenum.str()+".maxval";
  write_velocity_response(tempPath+"/"+refmax,max_response,pic_size,offset);

  dealloc2ddoublearray(max_response,pic_size[0],pic_size[1]);

  delete [] size;
  delete [] sigma;
  delete [] radius;

}


/*********************************************************************/
/* Filters the images with the set of Gabor filters and tries to use */
/* what it can of available filtered reference images                */
/*********************************************************************/

void filter_images(double*** ref_img,double ***dis_img,string tempPath,string filtPath,string outputPath,string refname,string disname,int* pic_size,int num_scales,int central_image)
{
  dcmplx *gabor_filter_x,*gabor_filter_y,*gabor_filter_tt;
  double *gauss_filter_x,*gauss_filter_y,*gauss_filter_tt;
  dcmplx **gabor_output;
  double **gauss_output;
  double **max_response;
  double **filter_centerfreq;
  int CANNOT_COMPUTE_MAX = FALSE;

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];

  double maxamp = 0;

  /* Compute the center frequencies of the Gabor filters */
  initFilterParameters(num_scales,size,radius,sigma);
  filter_centerfreq = init_filters(radius,num_scales);

  /* Set DC parameters */

  double sigma_dc_freq = radius[num_scales-1]-1/sigma[num_scales-1];
  double sigma_dc = 1.0/sigma_dc_freq;
  int size_dc = static_cast<int>(6*sigma_dc + 1);
  if(size_dc%2 == 0){ 
    size_dc += 1;
  }

  const double dc_factor = 0.001;  
  int len,offset;
  int filter_size[3];
  int found_ref_file,found_dis_file;

  /* Declare variables to hold the output */

  gabor_output = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  gauss_output = alloc2ddoublearray(pic_size[0],pic_size[1]);
  max_response = alloc2ddoublearray(pic_size[0],pic_size[1]);
  initialize_values(max_response,pic_size[0],pic_size[1],0,0,0);

  ostringstream centralFramenum;
  centralFramenum << central_image;

  len = size_dc;
  offset = size_dc/2;
  filter_size[0] = filter_size[1] = filter_size[2] = len;
  cout << "Convolving video with DC filter:" << endl;
  
  gauss_filter_x = new double[len];
  gauss_filter_y = new double[len];
  gauss_filter_tt = new double[len];

  string refdc = refname + ".frame" + centralFramenum.str()+".dc";
  string disdc = disname + ".frame" + centralFramenum.str()+".dc";

  create_gauss_1d(gauss_filter_x,sigma_dc,len);
  create_gauss_1d(gauss_filter_y,sigma_dc,len);
  create_gauss_1d(gauss_filter_tt,sigma_dc,len);

  separable_convolve(gauss_output,ref_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);
  write_velocity_response(tempPath+"/"+refdc,gauss_output,pic_size,offset);

  separable_convolve(gauss_output,dis_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);
  write_velocity_response(tempPath+"/"+disdc,gauss_output,pic_size,offset);

  delete [] gauss_filter_x;
  delete [] gauss_filter_y;
  delete [] gauss_filter_tt;

  cout << "Convolving video with Gabor filters: ";
  for(int i=0; i<num_scales; i++){

    cout << "\nScale " << i << ": \n";
    ostringstream scalenum;
    scalenum << i;

    len = size[i];
    offset = size[i]/2;
    filter_size[0] = filter_size[1] = filter_size[2] = len;

    /* Declare variables to hold the filter */

    gabor_filter_x = new dcmplx[len];
    gabor_filter_y = new dcmplx[len];
    gabor_filter_tt = new dcmplx[len];

    gauss_filter_x = new double[len];
    gauss_filter_y = new double[len];
    gauss_filter_tt = new double[len];

    for(int j=0; j<NFILTERS; j++){

      fflush(stdout);
      int filtnum = i*NFILTERS+j;


      for(int frameind=central_image;frameind <= central_image;frameind++){
	ostringstream filenum;
	filenum << filtnum;


	ostringstream framenum;
	framenum << frameind;


	string refbase = refname + ".scale" + scalenum.str() + ".frame" + framenum.str()+".gab"+filenum.str();
	string disbase = disname + ".scale" + scalenum.str() + ".frame" + framenum.str()+".gab"+filenum.str();

	string refout = filtPath + "/" + refbase;
	ifstream in(refout.c_str(),ios::in|ios::binary);

	if(!in){
	  in.close();
	  found_ref_file = FALSE;
	}
	else{
	  found_ref_file = TRUE;
	  CANNOT_COMPUTE_MAX = TRUE;
	  in.close();
	}

	string disout = filtPath + "/" + disbase;
	in.open(disout.c_str(),ios::in|ios::binary);

	if(!in){
	  in.close();
	  found_dis_file = FALSE;
	}
	else{
	  found_dis_file = TRUE;
	  in.close();
	}

	if(!found_ref_file || !found_dis_file){

	  cout << i*NFILTERS+j << "\t" ;

	  /* Create the Gabor filter */
	  create_gabor_1d(gabor_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_gabor_1d(gabor_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_gabor_1d(gabor_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);

	  create_gauss_1d(gauss_filter_x,sigma[i],len);
	  create_gauss_1d(gauss_filter_y,sigma[i],len);
	  create_gauss_1d(gauss_filter_tt,sigma[i],len);

	  if(!found_ref_file){
	    separable_convolve(gabor_output,ref_img,gabor_filter_x,gabor_filter_y,gabor_filter_tt,pic_size,len,len,len);
	    separable_convolve(gauss_output,ref_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);
	    
	    /* Subtract 0.001 of the gaussian from the cosine filter to remove dc component */
	    subImages(gabor_output,gauss_output,pic_size,dc_factor);
	    
	    /* Write the outputs to a file */
	    write_filtered_files(tempPath,refbase,pic_size,offset,gabor_output);
	  
	    double curr_max = calc_maxamp(gabor_output,pic_size,max_response);
	    if(maxamp < curr_max){
	      maxamp = curr_max;
	    }
	  }

	  if(!found_dis_file){
	    separable_convolve(gabor_output,dis_img,gabor_filter_x,gabor_filter_y,gabor_filter_tt,pic_size,len,len,len);
	    separable_convolve(gauss_output,dis_img,gauss_filter_x,gauss_filter_y,gauss_filter_tt,pic_size,len,len,len);
	    
	    /* Subtract 0.001 of the gaussian from the cosine filter to remove dc component */
	    subImages(gabor_output,gauss_output,pic_size,dc_factor);
	    
	    /* Write the outputs to a file */
	    write_filtered_files(tempPath,disbase,pic_size,offset,gabor_output);
	  }
	}
      }
    }

    /* Deallocate all variables */

    delete [] gabor_filter_x;
    delete [] gabor_filter_y;
    delete [] gabor_filter_tt;
    delete [] gauss_filter_x;
    delete [] gauss_filter_y;
    delete [] gauss_filter_tt;
  } 

  dealloc2dcomplexarray(gabor_output,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(gauss_output,pic_size[0],pic_size[1]);

  /* Deallocate the filter_centerfreq variable */

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }

  delete [] filter_centerfreq; 

  if(!found_ref_file){
    /* Write the max value to a file */

    string s = refname + ".frame" + centralFramenum.str() + ".max";  
    write_max_value(tempPath,s,maxamp);
  }

  if(!CANNOT_COMPUTE_MAX){
    offset = size[num_scales-1]/2;
    string refmax = refname + ".frame" + centralFramenum.str()+".maxval";
    write_velocity_response(tempPath+"/"+refmax,max_response,pic_size,offset);
  }

  dealloc2ddoublearray(max_response,pic_size[0],pic_size[1]);

  delete [] size;
  delete [] sigma;
  delete [] radius;

}

/*********************************************************************/
/* Read the outputs of the Gabor filters from a file                 */
/*********************************************************************/

void read_filtered_files(string filterPath, string filename,int *pic_size, int& offset, dcmplx **filt_output){

  ifstream in;

  string filein = filterPath + "/" + filename;
  in.open(filein.c_str(),ios::in|ios::binary);
  assert(in);

  int nrows, ncols, off;
  in.read(reinterpret_cast<char*> (&nrows),sizeof(int));
  in.read(reinterpret_cast<char*> (&ncols),sizeof(int));
  in.read(reinterpret_cast<char*> (&off),sizeof(int));

  pic_size[0] = nrows;
  pic_size[1] = ncols;
  offset = off;

  for(int j=0; j<pic_size[0]; j++){
    in.read(reinterpret_cast<char*>(filt_output[j]),sizeof(dcmplx)*pic_size[1]);
  }

  in.close();
}

/*********************************************************************/
/* Write the outputs of the Gabor filters into a file                */
/*********************************************************************/

void write_filtered_files(string filterPath, string filename,int *pic_size, int offset, dcmplx **filt_output){

  ofstream out;
  string fileout = filterPath + "/" + filename;
  out.open(fileout.c_str(),ios::out|ios::binary);

  assert(out);

  int nrows = pic_size[0];
  int ncols = pic_size[1];
  int off = offset;

  /* First, write the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  out.write(reinterpret_cast<char*>(&nrows), sizeof(int));
  out.write(reinterpret_cast<char*>(&ncols),sizeof(int));
  out.write(reinterpret_cast<char*>(&off),sizeof(int));

  for(int j=0; j<pic_size[0]; j++){
    out.write(reinterpret_cast<char*>(filt_output[j]),sizeof(dcmplx)*pic_size[1]);
  }

  out.close();
}

/****************************************************************************************************************************************/
/*  Initializes values in movie_values to val1 if not boundary and val2 if in the boundary. Boundary is decided by offset.               */
/****************************************************************************************************************************************/

void initialize_values(dcmplx **movie_values, int nrows, int ncols, int offset, dcmplx val1, dcmplx val2)
{
  int i,j;

  for(i=0;i<nrows;i++){
    for(j=0;j<ncols;j++){
      if((i<offset) | (i>nrows-offset) | (j<offset) | (j>ncols-offset))
	movie_values[i][j] = val1;
      else
	movie_values[i][j] = val2;
    }
  }
}

/**************************************************************************/
/*               Initialize values (overloaded)                           */
/**************************************************************************/

void initialize_values(double **movie_values, int nrows, int ncols, int offset, double val1, double val2)
{
  int i,j;

  for(i=0;i<nrows;i++){
    for(j=0;j<ncols;j++){
      if((i<offset) | (i>nrows-offset) | (j<offset) | (j>ncols-offset))
	movie_values[i][j] = val1;
      else
	movie_values[i][j] = val2;
    }
  }
}

/**************************************************************************/
/*               Initialize values                                   */
/**************************************************************************/

void initialize_values(unsigned char **movie_values, int nrows, int ncols, int offset, unsigned char val1, unsigned char val2)
{
  int i,j;

  for(i=0;i<nrows;i++){
    for(j=0;j<ncols;j++){
      if((i<offset) | (i>nrows-offset) | (j<offset) | (j>ncols-offset))
	movie_values[i][j] = val1;
      else
	movie_values[i][j] = val2;
    }
  }
}

/**************************************************************************/
/*               Initialize values                                   */
/**************************************************************************/

void initialize_values(double ***movie_values, int nrows, int ncols, int nframes,int offset, unsigned char val1, unsigned char val2)
{
  int i,j,k;

  for(k=0; k<nframes; k++){
    for(i=0;i<nrows;i++){
      for(j=0;j<ncols;j++){
	if((i<offset) | (i>nrows-offset) | (j<offset) | (j>ncols-offset))
	  movie_values[k][i][j] = val1;
	else
	  movie_values[k][i][j] = val2;
      }
    }
  }
}

/**************************************************************************/
/*      Computes the distance of the plane from the center frequency      */
/*      of the filter                                                     */
/**************************************************************************/

double compute_dist_from_plane(double vx,double vy,double *filter_centerfreq){

  double norm_term = sqrt(pow(vx,2)+pow(vy,2)+1);

  double wx0 = filter_centerfreq[0];
  double wy0 = filter_centerfreq[1];
  double wt0 = filter_centerfreq[2];

  double dist = (vx*wx0+vy*wy0+wt0)/norm_term;

  return dist;

}

/*************************************************************************/
/*           Write MOVIE maps to file                               */
/*************************************************************************/

void write_movie_maps(string fileout,double **movie_values,int *pic_size, int offset){

  ofstream out;
  if(PRINT){  
    cout << "Writing MOVIE values to file " << fileout << endl;
  }

  out.open(fileout.c_str(),ios::out|ios::binary);
  assert(out);
  int nrows = pic_size[0];
  int ncols = pic_size[1];
  int off = offset;

  /* First, write the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  out.write(reinterpret_cast<char*>(&nrows), sizeof(int));
  out.write(reinterpret_cast<char*>(&ncols),sizeof(int));
  out.write(reinterpret_cast<char*>(&off),sizeof(int));

  for(int j=0; j<pic_size[0]; j++){
    out.write(reinterpret_cast<char*>(movie_values[j]),sizeof(double)*pic_size[1]);
  }

  out.close();

}

/*************************************************************************/
/*           Write MOVIE values to file                               */
/*************************************************************************/

void write_movie_values(string fileout,double *movie_values,int pic_size){

  ofstream out;
  if(PRINT){  
    cout << "Writing MOVIE values to file " << fileout << endl;
  }

  out.open(fileout.c_str(),ios::out);
  assert(out);

  for(int j=0; j<pic_size; j++){
    ostringstream s;
    s << movie_values[j];
    out << s.str() << "\n";
  }

  out.close();

}
void write_movie_values(string fileout,double& movie_values){

  ofstream out;
  if(PRINT){  
    cout << "Writing MOVIE values to file " << fileout << endl;
  }

  out.open(fileout.c_str(),ios::out);
  assert(out);

  ostringstream s;
  s << movie_values;
  out << s.str() << "\n";

  out.close();

}

/*************************************************************************/
/*           Read MOVIE values from file                               */
/*************************************************************************/

void read_movie_maps(string fileout,double **movie_values,int *pic_size, int& offset){

  ifstream in;
  if(PRINT){  
    cout << "Reading MOVIE maps from file " << fileout << endl;
  }

  in.open(fileout.c_str(),ios::in|ios::binary);
  assert(in);
  int nrows,ncols,off;

  /* First, write the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  in.read(reinterpret_cast<char*>(&nrows), sizeof(int));
  in.read(reinterpret_cast<char*>(&ncols),sizeof(int));
  in.read(reinterpret_cast<char*>(&off),sizeof(int));

  pic_size[0] = nrows;
  pic_size[1] = ncols;
  offset = off;

  for(int j=0; j<nrows; j++){
    in.read(reinterpret_cast<char*>(movie_values[j]),sizeof(double)*pic_size[1]);
  }

  in.close();

}

/***************************************************************************/
/*             Implements separable convolution for complex filters        */
/***************************************************************************/

void separable_convolve(dcmplx **gabor_output,double ***ref_img,dcmplx *filter_x,dcmplx *filter_y,dcmplx *filter_tt,int *pic_size,int filter_x_size,int filter_y_size,int filter_t_size){

  dcmplx **filtered_tt = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  conv_tt(filtered_tt,ref_img,filter_tt,pic_size,filter_t_size);

  dcmplx **filtered_xt = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  conv_x(filtered_xt,filtered_tt,filter_x,pic_size,filter_x_size);
  dealloc2dcomplexarray(filtered_tt,pic_size[0],pic_size[1]);

  conv_y(gabor_output,filtered_xt,filter_y,pic_size,filter_y_size);
  dealloc2dcomplexarray(filtered_xt,pic_size[0],pic_size[1]);

}

void conv_tt(dcmplx **gabor_output,double ***ref_img,dcmplx *filter_tt,int *pic_size,int filter_t_size){

  int middle = pic_size[2]/2; //Center frame of the video
  int offset = filter_t_size/2;

  initialize_values(gabor_output,pic_size[0],pic_size[1],offset,0,0);

  for(int i=0; i<pic_size[0]; i++){//rows
    for(int j=0; j<pic_size[1];j++){//columns

      for(int m=0; m<filter_t_size; m++){
	gabor_output[i][j] += (ref_img[middle-offset+m][i][j]*filter_tt[filter_t_size-1-m]);
      }
    }
  }
}

void conv_x(dcmplx **gabor_output,dcmplx **ref_img,dcmplx *filter_x,int *pic_size,int filter_x_size){

  int offset = filter_x_size/2;

  initialize_values(gabor_output,pic_size[0],pic_size[1],offset,0,0);

  for(int i=offset; i<pic_size[0]-offset;i++){//rows
    for(int j=0; j<pic_size[1];j++){//columns

      for(int m=0; m<filter_x_size; m++){
	gabor_output[i][j] += (ref_img[i-offset+m][j]*filter_x[filter_x_size-1-m]);
      }
    }
  }
}

void conv_y(dcmplx **gabor_output,dcmplx **ref_img,dcmplx *filter_y,int *pic_size,int filter_y_size){

  int offset = filter_y_size/2;

  initialize_values(gabor_output,pic_size[0],pic_size[1],offset,0,0);

  for(int i=0; i<pic_size[0];i++){//rows
    for(int j=offset; j<pic_size[1]-offset;j++){//columns

      for(int m=0; m<filter_y_size; m++){
	gabor_output[i][j] += (ref_img[i][j-offset+m]*filter_y[filter_y_size-1-m]);
      }
    }
  }
}


/***************************************************************************/
/*             Implements separable convolution for double filters        */
/***************************************************************************/

void separable_convolve(double **gabor_output,double ***ref_img,double *filter_x,double *filter_y,double *filter_tt,int *pic_size,int filter_x_size,int filter_y_size,int filter_t_size){

  double **filtered_tt = alloc2ddoublearray(pic_size[0],pic_size[1]);
  conv_tt(filtered_tt,ref_img,filter_tt,pic_size,filter_t_size);

  double **filtered_xt = alloc2ddoublearray(pic_size[0],pic_size[1]);
  conv_x(filtered_xt,filtered_tt,filter_x,pic_size,filter_x_size);
  dealloc2ddoublearray(filtered_tt,pic_size[0],pic_size[1]);

  conv_y(gabor_output,filtered_xt,filter_y,pic_size,filter_y_size);
  dealloc2ddoublearray(filtered_xt,pic_size[0],pic_size[1]);

}

void conv_x(double **gabor_output,double **ref_img,double *filter_x,int *pic_size,int filter_x_size){

  int offset = filter_x_size/2;

  initialize_values(gabor_output,pic_size[0],pic_size[1],offset,0,0);

  for(int i=offset; i<pic_size[0]-offset;i++){//rows
    for(int j=0; j<pic_size[1];j++){//columns

      for(int m=0; m<filter_x_size; m++){
	gabor_output[i][j] += (ref_img[i-offset+m][j]*filter_x[filter_x_size-1-m]);
      }
    }
  }
}

void conv_y(double **gabor_output,double **ref_img,double *filter_y,int *pic_size,int filter_y_size){

  int offset = filter_y_size/2;

  initialize_values(gabor_output,pic_size[0],pic_size[1],offset,0,0);

  for(int i=0; i<pic_size[0];i++){//rows
    for(int j=offset; j<pic_size[1]-offset;j++){//columns

      for(int m=0; m<filter_y_size; m++){
	gabor_output[i][j] += (ref_img[i][j-offset+m]*filter_y[filter_y_size-1-m]);
      }
    }
  }
}

void conv_tt(double **gabor_output,double ***ref_img,double *filter_tt,int *pic_size,int filter_t_size){

  int middle = pic_size[2]/2; //Center frame of the video
  int offset = filter_t_size/2;

  initialize_values(gabor_output,pic_size[0],pic_size[1],offset,0,0);

  for(int i=0; i<pic_size[0];i++){//rows
    for(int j=0; j<pic_size[1];j++){//columns

      for(int m=0; m<filter_t_size; m++){
	gabor_output[i][j] += (ref_img[middle-offset+m][i][j]*filter_tt[filter_t_size-1-m]);
      }
    }
  }
}

/***************************************************************************/
/*             Implements separable convolution for double filters - 2D    */
/***************************************************************************/

void separable_convolve(double **gabor_output,double **ref_img,double *filter_x,double *filter_y,int *pic_size,int filter_x_size,int filter_y_size){

  double **filtered_x = alloc2ddoublearray(pic_size[0],pic_size[1]);
  conv_x(filtered_x,ref_img,filter_x,pic_size,filter_x_size);

  conv_y(gabor_output,filtered_x,filter_y,pic_size,filter_y_size);
  dealloc2ddoublearray(filtered_x,pic_size[0],pic_size[1]);

}

/*****************************************************************/
/* Computes the maximum filter amplitude                         */
/*****************************************************************/

double calc_maxamp(dcmplx **filt_output,int *pic_size, double **max_response){

  double max = 0;
  for(int i=0; i<pic_size[0]; i++){
    for(int j=0; j<pic_size[1]; j++){
      double amp = abs(filt_output[i][j]);
      if(max < amp)
	max = amp;
      if(max_response[i][j] < amp){
	max_response[i][j] = amp;
      }
    }
  }
  return max;
}

/*****************************************************************/
/* Computes the maximum filter amplitude                         */
/*****************************************************************/

double calc_maxamp(double **filt_output,int *pic_size, double **max_response){

  double max = 0;
  for(int i=0; i<pic_size[0]; i++){
    for(int j=0; j<pic_size[1]; j++){
      double amp = abs(filt_output[i][j]);
      if(max < amp)
	max = amp;
      if(max_response[i][j]<amp){
	max_response[i][j] = amp;
      }
    }
  }
  return max;
}

/*****************************************************************/
/*        Read and write the maximum filter response                    */
/*****************************************************************/

double read_max_value(string filterPath,string s){

  ifstream in;
  string filename = filterPath + "/" + s;  
  in.open(filename.c_str(),ios::in | ios::binary);
  assert(in);
  double maxamp;

  in.read(reinterpret_cast<char*> (&maxamp),sizeof(double));
  in.close();

  if(PRINT){
    cout << "Maximum amplitude: " << maxamp << endl;
  }
  return maxamp;
}

void write_max_value(string tempPath, string s, double maxamp){

  ofstream out;
  string filename = tempPath + "/" + s;  

  out.open(filename.c_str(),ios::out | ios::binary);
  assert(out);

  out.write(reinterpret_cast<char*> (&maxamp), sizeof(double));
  out.close();  
}

/********************************************************************************/
/* Creates the Gabor filters ****************************************************/
/********************************************************************************/

void create_gabor_1d(dcmplx *filter,double center_freq,double sigma,int len){

  int offset = len/2;
  double gauss;
  const double scale = 1/(pow(2*PI,0.5)*sigma);

  /* Make sure you don't go out of bounds */

  if(len!=(2*offset+1)){
    cout << "Error: The filter size is not odd " << endl;
    exit(-1);
  }

  for(int k=-offset; k<=offset; k++){

    gauss = scale*exp(-pow(static_cast<double>(k),2)/(2*pow(sigma,2)));
    dcmplx temp(gauss*cos(center_freq*k),gauss*sin(center_freq*k));
    filter[k+offset] = temp;
  }
}

/******************************************************************/
/*      Creates Gaussian filter to subtract dc component          */
/******************************************************************/

void create_gauss_1d(double *filter,double sigma,int len){
  int offset = len/2;
  double gauss;
  const double scale = 1/(pow(2*PI,0.5)*sigma);

  /* Make sure you don't go out of bounds */

  if(len!=(2*offset+1)){
    cout << "Error: The filter size is not odd " << endl;
    exit(-1);
  }

  for(int k=-offset; k<=offset; k++){

    gauss = exp(-pow(static_cast<double>(k),2)/(2*pow(sigma,2)));
    filter[k+offset] = scale*gauss;
  }
}


/****************************************************************/
/* Subtracts 2 images after multiplying one with a scale factor */
/****************************************************************/

void subImages(dcmplx** img1, double** img2, int* pic_size, double alpha)
{
  for(int j=0; j<pic_size[0]; j++){
    for(int k=0; k<pic_size[1]; k++){
      img1[j][k] -= alpha*img2[j][k];
    }
  } 
}

/****************************************************************/
/* Subtracts 2 images after multiplying one with a scale factor */
/****************************************************************/

void subImages(double** img1, double** img2, int* pic_size)
{
  for(int j=0; j<pic_size[0]; j++){
    for(int k=0; k<pic_size[1]; k++){
      img1[j][k] -= img2[j][k];
    }
  } 
}

/*********************************************************************/
/* Write the velocity tuned responses into a file                */
/*********************************************************************/

void write_velocity_response(string filename,double **vel_response,int *vel_size, int offset){

  ofstream out(filename.c_str(),ios::out|ios::binary);
  assert(out);

  int nrows = vel_size[0];
  int ncols = vel_size[1];
  int off = offset;

  /* First, write the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  out.write(reinterpret_cast<char*>(&nrows), sizeof(int));
  out.write(reinterpret_cast<char*>(&ncols),sizeof(int));
  out.write(reinterpret_cast<char*>(&off),sizeof(int));

  for(int i=0; i<nrows; i++){
    out.write(reinterpret_cast<char*>(vel_response[i]),sizeof(double)*ncols);
  }  

  out.close();

}

/*********************************************************************/
/* Read the velocity tuned responses from a file                 */
/*********************************************************************/

void read_velocity_response(string filename,double **vel_response,int *vel_size, int& offset){

  ifstream in(filename.c_str(),ios::in|ios::binary);
  assert(in);
  
  int nrows, ncols,off;
  in.read(reinterpret_cast<char*> (&nrows),sizeof(int));
  in.read(reinterpret_cast<char*> (&ncols),sizeof(int));
  in.read(reinterpret_cast<char*> (&off),sizeof(int));
  
  for(int i=0; i<nrows; i++){
    in.read(reinterpret_cast<char*>(vel_response[i]),sizeof(double)*ncols);
  }
  
  in.close();
}

/*********************************************************************************/
/*     Thresholds and performs velocity computation on the filtered images       */
/*********************************************************************************/

void thresh_and_compute(string filename,string filterPath,string outputPath,int *pic_size,int central_image,int num_scales,double tau, double percent_maxamp){

  unsigned char ***thresholded;
  double ***full_velocities, ****full_velocities_all_scales;
  dcmplx ***normal_velocities; //Just store the velocities in a complex variable 
  ostringstream framenum;

  thresholded = alloc3dchararray(pic_size[0],pic_size[1],num_scales*NFILTERS);
  normal_velocities = alloc3dcomplexarray(pic_size[0],pic_size[1],num_scales*NFILTERS);
 
  /* Initialize thresholds and full velocities */
  initialize_thresholds(thresholded,pic_size[0],pic_size[1],num_scales*NFILTERS);

  /* Threshold filter responses and compute normal velocities */
  cout << "Computing normal velocities...\n";
  threshold(filterPath,filename,pic_size,central_image,thresholded,normal_velocities,percent_maxamp,tau,num_scales);

  /* Compute full velocities */
  cout << "\nComputing 2D velocities...\n";

  full_velocities = alloc3ddoublearray(pic_size[0],pic_size[1],2);
  full_velocities_all_scales = new double***[num_scales];
  for (int ii=0; ii<num_scales; ii++){
    full_velocities_all_scales[ii] = alloc3ddoublearray(pic_size[0],pic_size[1],2);
  }

  initialize_full_velocities(full_velocities,pic_size[0],pic_size[1],2);
  for (int ii=0; ii<num_scales; ii++){
    initialize_full_velocities(full_velocities_all_scales[ii],pic_size[0],pic_size[1],2);
  }
   
  compute_full_velocities(full_velocities_all_scales,full_velocities,normal_velocities,thresholded,pic_size,num_scales);

  /* Output full velocities */

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  initFilterParameters(num_scales,size,radius,sigma);
  int offset = size[0]/2;

  /* Create filenames */

  framenum << central_image;
  string s = filename + ".frame" + framenum.str() + ".finalflow";
  output_full_velocities(filterPath,s.c_str(),pic_size,offset,full_velocities);

  /* Output full velocities at all scales */
  for(int ii=0; ii<num_scales; ii++){
    offset = size[ii]/2;
    ostringstream scalenum;
    scalenum << ii;
    s = filename + ".scale" + scalenum.str() + ".frame" + framenum.str() + ".flow";
    output_full_velocities(filterPath,s.c_str(),pic_size,offset,full_velocities_all_scales[ii]);
  }

  offset = size[0]/2;

  dealloc3dchararray(thresholded,pic_size[0],pic_size[1],num_scales*NFILTERS);
  dealloc3ddoublearray(full_velocities,pic_size[0],pic_size[1],2);
  dealloc3dcomplexarray(normal_velocities,pic_size[0],pic_size[1],num_scales*NFILTERS);
  for (int ii=0; ii<num_scales; ii++){
    dealloc3ddoublearray(full_velocities_all_scales[ii],pic_size[0],pic_size[1],2);
  }

  delete [] full_velocities_all_scales;
  delete [] size;
  delete [] radius;
  delete []  sigma;

}

/**************************************************************************/
/*               Initialize thresholds                                    */
/**************************************************************************/

void initialize_thresholds(unsigned char ***thresholded, int nrows, int ncols, int nframes)
{
  int i,j,k;

  for(i=0;i<nframes;i++)
    {
      for(j=0;j<nrows;j++){
	for(k=0;k<ncols;k++){
	  thresholded[i][j][k] = NOT_THRESHOLDED;
	}
      }
    }
  if(PRINT){
    printf("Thresholds initialized to NOT_THRESHOLDED everywhere \n");
  }
}

/**************************************************************************/
/*               Initialize full velocities                               */
/**************************************************************************/

void initialize_full_velocities(double ***full_velocities, int nrows, int ncols, int nframes)
{
  int i,j,k;

  for(i=0;i<nframes;i++)
    {
      for(j=0;j<nrows;j++){
	for(k=0;k<ncols;k++){
	  full_velocities[i][j][k] = UNDEFINED;
	}
      }
    }
  if(PRINT){
    printf("Full velocity initialized to UNDEFINED everywhere \n");
  }
}

/***************************************************************************/
/* Threshold the derivative filtered results and compute normal velocities */
/***************************************************************************/

void threshold(string filterPath,string filename,int *pic_size,int central_image,unsigned char ***thresholded,dcmplx ***normal_velocities,double percent_maxamp,double tau,int num_scales){

  ostringstream framenum;
  framenum << central_image;

  double maxamp;
  dcmplx **gab,**gabx,**gaby,**gabt;
  long int* num_ampthr = new long int[num_scales*NFILTERS];
  long int* num_freqthr = new long int[num_scales*NFILTERS];
  long int* unthr = new long int[num_scales*NFILTERS];
  int offset;

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  double **filter_centerfreq;
  double lhsterm,rhsterm,mag2,dphix,dphiy,dphit; 
  static int test = 0;

  /* Compute the center frequencies of the Gabor filters */
  initFilterParameters(num_scales,size,radius,sigma);
  filter_centerfreq = init_filters(radius,num_scales);

  /* Read in the max value from file */

  string s = filename + ".frame" + framenum.str() + ".max";  
  maxamp = read_max_value(filterPath,s);
  if(PRINT){
    cout << "Maximum amplitude: " << maxamp << endl;
    cout << "Maximum amplitude threshold percentage: " << percent_maxamp*100 << endl;
    cout << "Thresold value: " << percent_maxamp*maxamp << endl;
  }

  const double ampthresh2 = pow(percent_maxamp*maxamp,2);

  gab = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  gabx = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  gaby = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  gabt = alloc2dcomplexarray(pic_size[0],pic_size[1]);

  for(int i=0; i<num_scales; i++){
    ostringstream scalenum;
    scalenum << i;
    cout << "\nScale " << scalenum.str() << ": "; 
    for(int j=0; j<NFILTERS; j++){

      int filtnum = i*NFILTERS+j;
      cout << filtnum << "\t";
      double sig = sigma[i];
      num_ampthr[filtnum] = 0;
      num_freqthr[filtnum] = 0;
      unthr[filtnum] = 0;
      long int count = 0;

      /* Read in the Gabor outputs */

      ostringstream filenum;
      filenum << filtnum;

      string imgbase = filename + ".scale" + scalenum.str() + ".frame" + framenum.str()+".gab"+filenum.str();
      read_filtered_files(filterPath,imgbase,pic_size,offset,gab);

      /* Read in the computed derivatives */

      imgbase = filename + ".scale" + scalenum.str() + ".frame" + framenum.str() + ".gabx" + filenum.str();	  
      read_filtered_files(filterPath,imgbase,pic_size,offset,gabx);
      imgbase = filename + ".scale" + scalenum.str() + ".frame" + framenum.str() + ".gaby" + filenum.str();	  
      read_filtered_files(filterPath,imgbase,pic_size,offset,gaby);
      imgbase = filename + ".scale" + scalenum.str() + ".frame" + framenum.str() + ".gabt" + filenum.str();	  
      read_filtered_files(filterPath,imgbase,pic_size,offset,gabt);

      if(PRINT && test==0){

	cout << "\n\n\n Central frame of the Gabor filtered outputs of filter 1: \n";
	for(int ii=0; ii<pic_size[0]; ii++){
	  for(int jj=0; jj<pic_size[1]; jj++){
	    cout << gab[ii][jj] << " ";
	  }
	  cout << endl;
	}

	cout << "\n\n\n Central frame of the Gabor filtered outputs of x derivative of filter 1: \n";
	for(int ii=0; ii<pic_size[0]; ii++){
	  for(int jj=0; jj<pic_size[1]; jj++){
	    cout << gabx[ii][jj] << " ";
	  }
	  cout << endl;
	}
	cout << "\n\n\n Central frame of the Gabor filtered outputs of y derivative of filter 1: \n";
	for(int ii=0; ii<pic_size[0]; ii++){
	  for(int jj=0; jj<pic_size[1]; jj++){
	    cout << gaby[ii][jj] << " ";
	  }
	  cout << endl;
	}
	cout << "\n\n\n Central frame of the Gabor filtered outputs t derivative of filter 1: \n";
	for(int ii=0; ii<pic_size[0]; ii++){
	  for(int jj=0; jj<pic_size[1]; jj++){
	    cout << gabt[ii][jj] << " ";
	  }
	  cout << endl;
	}
	test = 1;
      }

      double rrx,iix,realx2,imagx2,rry,iiy,realy2,imagy2,rrt,iit,realt2,imagt2;
      dcmplx rdelrx,rdelry,rdelrt;

      for(int m=offset; m<pic_size[0]-offset; m++){
	for(int n=offset; n<pic_size[1]-offset; n++){
	  count++;
	  mag2 = pow(abs(gab[m][n]),2);
	  if(mag2!=0.0){
	    rrx = gab[m][n].real()*gabx[m][n].real();
	    iix = gab[m][n].imag()*gabx[m][n].imag();
	    realx2 = pow((rrx+iix)/mag2,2);
	    rdelrx = conj(gab[m][n])*gabx[m][n];
	    dphix = rdelrx.imag()/mag2;
	    imagx2 = pow(dphix-filter_centerfreq[filtnum][0],2);
	    rry = gab[m][n].real()*gaby[m][n].real();
	    iiy = gab[m][n].imag()*gaby[m][n].imag();
	    realy2 = pow((rry+iiy)/mag2,2);
	    rdelry = conj(gab[m][n])*gaby[m][n];
	    dphiy = rdelry.imag()/mag2;
	    imagy2 = pow(dphiy-filter_centerfreq[filtnum][1],2);
	    rrt = gab[m][n].real()*gabt[m][n].real();
	    iit = gab[m][n].imag()*gabt[m][n].imag();
	    realt2 = pow((rrt+iit)/mag2,2);
	    rdelrt = conj(gab[m][n])*gabt[m][n];
	    dphit = rdelrt.imag()/mag2;
	    imagt2 = pow(dphit-filter_centerfreq[filtnum][2],2);

	    lhsterm = realx2+realy2+realt2+imagx2+imagy2+imagt2;

	    /* Note that sig is the std. dev. of the Gabor in space, so we use 1/sig in the formula here */
	    rhsterm = pow(tau/sig,2);
	  }
	  else{
	    lhsterm = HUGE;
	  }

	  if(mag2<ampthresh2){
	    num_ampthr[filtnum]++;
	    thresholded[filtnum][m][n] = THRESHOLDED;
	    dcmplx temp(UNDEFINED,UNDEFINED);
	    normal_velocities[filtnum][m][n] = temp;
	  }
	  else if(lhsterm > rhsterm){
	    num_freqthr[filtnum]++;
	    thresholded[filtnum][m][n] = THRESHOLDED;
	    dcmplx temp(UNDEFINED,UNDEFINED);
	    normal_velocities[filtnum][m][n] = temp;
	  }

	  else if(mag2>ampthresh2 && lhsterm < rhsterm){
	    unthr[filtnum]++;
	    double denom = pow(dphix,2)+pow(dphiy,2);
	    double vx = -(dphit*dphix)/denom;
	    double vy = -(dphit*dphiy)/denom;
	    dcmplx v(vx,vy);
	    normal_velocities[filtnum][m][n] = v;
	  }
	}
      }
    }

    long int tot = (pic_size[0]-2*offset)*(pic_size[1]-2*offset);
    if(PRINT){		
      produce_thresh_report(num_ampthr,num_freqthr,unthr,i,tot);
    }

  }

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }
  delete [] filter_centerfreq;

  dealloc2dcomplexarray(gab,pic_size[0],pic_size[1]);
  dealloc2dcomplexarray(gabx,pic_size[0],pic_size[1]);
  dealloc2dcomplexarray(gaby,pic_size[0],pic_size[1]);
  dealloc2dcomplexarray(gabt,pic_size[0],pic_size[1]);

  delete [] size;
  delete [] radius;
  delete [] sigma;

  delete [] num_ampthr;
  delete [] num_freqthr;
  delete [] unthr;
}

/**************************************************************************/
/*              Produce thresholding report                               */
/**************************************************************************/

void produce_thresh_report(long int *num_ampthr,long int *num_freqthr,long int *unthr,int scalenum,long int tot){

  int filtnum;
  double total = static_cast<double>(tot);

  for(int j=0; j<NFILTERS; j++){

    filtnum = scalenum*NFILTERS+j;
    cout << "\n\nFilter " << filtnum << ": \n\n";

    cout << "Total number of responses: " << total << endl;
    cout << "Number of responses with amplitude < 5% of maximum: " << num_ampthr[filtnum] << endl;
    cout << "Percentage amplitude thresholded: " << (static_cast<double>(num_ampthr[filtnum])/total)*100 << "%\n";

    cout << "Number of responses with frequency outside tau*sigma of filter: " << num_freqthr[filtnum] << endl;
    cout << "Percentage frequency thresholded: " << (num_freqthr[filtnum]/total)*100 << "%\n";

    cout << "Number of responses left unthresholded: " << unthr[filtnum] << endl;
    cout << "Percentage not thresholded: " << (unthr[filtnum]/total)*100 << "%\n\n";		       
  }
}

/**************************************************************************/
/*              Compute full velocities for the frame                     */
/**************************************************************************/

void compute_full_velocities(double **** full_velocities_all_scales, double ***full_velocities,dcmplx ***normal_velocities,unsigned char ***thresholded,int *pic_size,int num_scales){

  int count,num_constraints;
  double n1,n2,mag,condnum;
  long int *num_condNumTooHigh, *num_errorTooHigh, *num_velComputed, *num_notEnoughConstraints,total_velComputed=0;
  num_condNumTooHigh = new long int[num_scales];
  num_velComputed = new long int[num_scales];
  num_errorTooHigh = new long int[num_scales];
  num_notEnoughConstraints = new long int[num_scales];

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  initFilterParameters(num_scales,size,radius,sigma);

  for(int ii=0; ii<num_scales; ii++){
    num_condNumTooHigh[ii] = num_errorTooHigh[ii] = num_velComputed[ii] = num_notEnoughConstraints[ii] = 0;
  }

  double v[3][3][2];
  double **J;
  int nrows = 0;
  int ncols = NUM_UNKNOWNS;
  double min_vel_error;
  int pixel_vel_computed_flag;

  for(int i=0; i<pic_size[0]; i++){
   
    for(int j=0; j<pic_size[1]; j++){

      pixel_vel_computed_flag = 0;
      min_vel_error = 1;

      for(int m=0; m<num_scales; m++){

	int offset = size[m]/2;
	if(i<offset || i>=pic_size[0]-offset || j<offset || j>=pic_size[1]-offset){
	  continue;
	}

	count = 0;
	nrows = 0;
	condnum = HUGE;

	for(int n=0; n<NFILTERS; n++){
	  int filtnum = m*NFILTERS+n;
	  for(int p=i-WINDOW; p<=i+WINDOW; p++){
	    for(int q=j-WINDOW; q<=j+WINDOW; q++){

	      if(thresholded[filtnum][p][q] == NOT_THRESHOLDED){
		nrows++;
	      }
	    }
	  }
	}

	if(nrows < NUM_UNKNOWNS){
	  num_notEnoughConstraints[m]++;
	  continue;
	}

	J = new double*[nrows];

	for(int ii=0; ii<nrows; ii++){
	  J[ii] = new double[ncols];
	}

	double *vn = new double[nrows];
	double *prod = new double[nrows];

	for(int n=0; n<NFILTERS; n++){
	  int filtnum = m*NFILTERS+n;
	  for(int p=i-WINDOW; p<=i+WINDOW; p++){
	    for(int q=j-WINDOW; q<=j+WINDOW; q++){
	      if(thresholded[filtnum][p][q] == NOT_THRESHOLDED){
		mag = abs(normal_velocities[filtnum][p][q]);
		if(mag > TOL){
		  n1 = normal_velocities[filtnum][p][q].real()/mag;
		  n2 = normal_velocities[filtnum][p][q].imag()/mag;
		  J[count][0] = n1;
		  J[count][1] = n1*(p-i);
		  J[count][2] = n1*(q-j);
		  J[count][3] = n2;
		  J[count][4] = n2*(p-i);
		  J[count][5] = n2*(q-j);
		  vn[count] = mag;
		  count++;
		}
	      }
	    }
	  }
	}


	num_constraints = count;
	if(num_constraints >= NUM_UNKNOWNS){
	  condnum = cal_velocity(num_constraints,J,vn,v,prod);
	}
	else{
	  condnum = HUGE;
	}

	double diff = 0;
	double vnNorm = 0;

	for(int ii=0; ii<count; ii++){
	  diff+=pow((prod[ii]-vn[ii]),2);
	  vnNorm += pow(vn[ii],2);
	}

	double vel_error = sqrt(diff)/sqrt(vnNorm);

	if(vel_error > 0.5){
	  num_errorTooHigh[m]++;
	}

	else if(condnum > 10)
	  num_condNumTooHigh[m]++;

	else if(vel_error < 0.5 && condnum < 10){

	  num_velComputed[m]++;
	  full_velocities_all_scales[m][0][i][j] = v[1][1][0];
	  full_velocities_all_scales[m][1][i][j] = v[1][1][1];

	  if(!pixel_vel_computed_flag){
	    total_velComputed++;
	    pixel_vel_computed_flag = 1;
	  }

	  if(vel_error < min_vel_error){
	    full_velocities[0][i][j] = v[1][1][0];
	    full_velocities[1][i][j] = v[1][1][1];
	    min_vel_error = vel_error;
	  }
	}

	/* Deallocate the memory allocated for J,vn and prod */

	for(int ii=0; ii<nrows; ii++)
	  delete [] J[ii];

	delete [] J;
	delete []vn;
	delete [] prod;

      }
    }
  }

  int offset;
  long int tot;
  double total;

  if(PRINTFLOW){
    for(int ii=0; ii<num_scales; ii++){
      offset = size[ii]/2;
      tot = (pic_size[0]-2*offset)*(pic_size[1]-2*offset);
      total = static_cast<double>(tot); 
      cout << "\n\nRESULTS FOR SCALE: " << ii << endl;

      cout << "Number of locations where full image velocity is computed: " << num_velComputed[ii] << ", ";
      cout << "Percentage: " << (num_velComputed[ii]/total)*100 << "% \n";
      
      cout << "Number of locations where condition number is too high: " << num_condNumTooHigh[ii] << ", ";
      cout << "Percentage: " << (num_condNumTooHigh[ii]/total)*100 << "% \n";
      
      cout << "Number of locations where error is too high: " << num_errorTooHigh[ii] << ", ";
      cout << "Percentage: " << (num_errorTooHigh[ii]/total)*100 << "% \n";
      
      cout << "Number of locations without enough constraints: " << num_notEnoughConstraints[ii] << ", ";
      cout << "Percentage: " << (num_notEnoughConstraints[ii]/total)*100 << "% \n";
    }

    offset = size[0]/2;
    tot = (pic_size[0]-2*offset)*(pic_size[1]-2*offset);
    total = static_cast<double>(tot); 
    
    cout << "\n\nFINAL TALLY: \n";
    cout << "Number of locations where full image velocity is computed: " << total_velComputed << ", ";
    cout << "Percentage: " << (total_velComputed/total)*100 << "% \n\n";

  }

  delete [] num_velComputed;
  delete [] num_condNumTooHigh;
  delete [] num_errorTooHigh;
  delete [] num_notEnoughConstraints;

  delete [] size;
  delete [] radius;
  delete [] sigma;

}

/**************************************************************************/
/*         Calculate velocity at a pixel                                  */
/**************************************************************************/

double cal_velocity(int num_constraints,double **J,double *vn,double v[3][3][2],double *prod){

  gsl_matrix *Jmatrix, *Vmatrix, *Jcopy;
  gsl_vector *S, *work, *x, *b;

  /* Make a copy of the original matrix since it gets replaced in the SVD computation */
  Jmatrix = gsl_matrix_alloc(num_constraints,NUM_UNKNOWNS);
  Jcopy = gsl_matrix_alloc(num_constraints,NUM_UNKNOWNS);
  if(Jmatrix==0 || Jcopy==0){
    cout << "Unable to allocate memory for J in cal_velocity \n";
  }

  for(int i=0; i<num_constraints; i++){
    for(int j=0; j<NUM_UNKNOWNS; j++){
      gsl_matrix_set(Jmatrix,i,j,J[i][j]);
      gsl_matrix_set(Jcopy,i,j,J[i][j]);
    }
  }

  Vmatrix = gsl_matrix_alloc(NUM_UNKNOWNS,NUM_UNKNOWNS);
  S = gsl_vector_alloc(NUM_UNKNOWNS);
  work = gsl_vector_alloc(NUM_UNKNOWNS);

  gsl_linalg_SV_decomp (Jmatrix,Vmatrix,S,work);

  /* Compute the condition number */

  double condnum,maxval,minval;
  maxval = minval =  gsl_vector_get(S,0);
  int num_singular_constraints = 0;

  for(int i=0; i<NUM_UNKNOWNS; i++){
    if(gsl_vector_get(S,i)>maxval && i!=0){
      cout << "Singular values are not increasing \n";
      exit(-1);
    }

    if((abs(gsl_vector_get(S,i)) > TOL) & (gsl_vector_get(S,i) < minval))
      minval = gsl_vector_get(S,i);

    if(abs(gsl_vector_get(S,i)) < TOL){
      gsl_vector_set(S,i,0);
      num_singular_constraints++;
    }
  }

  if(abs(minval) > TOL && (num_constraints-num_singular_constraints) > NUM_UNKNOWNS){
    condnum = maxval/minval;
  }
  else{ 
    condnum = HUGE;
    /* Deallocate all the gsl vectors */

    gsl_matrix_free(Jmatrix); gsl_matrix_free(Vmatrix); gsl_matrix_free(Jcopy);
    gsl_vector_free(S); gsl_vector_free(work);
    return condnum;
  }

  x = gsl_vector_alloc(NUM_UNKNOWNS);
  b = gsl_vector_alloc(num_constraints);

  for(int i=0; i<num_constraints; i++){
    gsl_vector_set(b,i,vn[i]);
  }

  gsl_linalg_SV_solve (Jmatrix,Vmatrix,S,b,x);

  /* Compute the prod vector using Jcopy since Jmatrix has been changed by SVD computation */

  for(int i=0; i<num_constraints; i++){
    prod[i] = 0;
    for(int j=0; j<NUM_UNKNOWNS; j++){
      prod[i] += gsl_matrix_get(Jcopy,i,j)*gsl_vector_get(x,j);
    }
  }

  /* Compute velocities in full neighborhood */

  for(int i=-1; i<=1; i++){
    for(int j=-1; j<=1; j++){
      v[i+1][j+1][0] = gsl_vector_get(x,0)+gsl_vector_get(x,1)*i+gsl_vector_get(x,2)*j;
      v[i+1][j+1][1] = gsl_vector_get(x,3)+gsl_vector_get(x,4)*i+gsl_vector_get(x,5)*j;
    }
  }

  /* Deallocate all the gsl vectors */

  gsl_matrix_free(Jmatrix); gsl_matrix_free(Vmatrix); gsl_matrix_free(Jcopy);
  gsl_vector_free(S); gsl_vector_free(work); gsl_vector_free(x); gsl_vector_free(b);

  return condnum;
}

/**************************************************************************/
/*                  Output full velocities                                */
/**************************************************************************/

void output_full_velocities(const string outputPath,const string filename,int *pic_size,int offset,double ***full_velocities){

  string s = outputPath + "/" + filename;

  ofstream out;
  out.open(s.c_str(),ios::out | ios::binary);
  assert(out);

  int nrows = pic_size[0];
  int ncols = pic_size[1];
  int off = offset;

  /* First, write the size of the flow field and then the offset, i.e, [nrows,ncols,offset] */

  out.write(reinterpret_cast<char*>(&nrows), sizeof(int));
  out.write(reinterpret_cast<char*>(&ncols),sizeof(int));
  out.write(reinterpret_cast<char*>(&off),sizeof(int));

  /* Write the flow values - u field followed by the v field*/

  for(int i=off; i<nrows-off; i++){
    out.write(reinterpret_cast<char*>(&full_velocities[0][i][off]),sizeof(double)*(ncols-2*off));
  }

  for(int i=off; i<nrows-off; i++){
    out.write(reinterpret_cast<char*>(&full_velocities[1][i][off]),sizeof(double)*(ncols-2*off));
  }

  out.close();

}

/**************************************************************************/
/*                 Create the derivative kernel along the x-direction     */
/**************************************************************************/

void create_derivative_kernel(dcmplx *filter, double center_freq,double sigma,int len){

  int offset = len/2;
  double gaussx,gauss;
  double sineterm;
  double cosineterm;
  const double scale = 1/(pow(2*PI,0.5)*sigma);
  static int test = 0;

  /* Make sure you don't go out of bounds */

  if(len!=(2*offset+1)){
    cout << "Error: The filter size is not odd " << endl;
    exit(-1);
  }

  for(int i=-offset; i<=offset; i++){
    gaussx = exp(-pow(static_cast<double>(i),2)/(2*pow(sigma,2)));
    gauss = scale*gaussx;

    sineterm = sin(center_freq*i);
    cosineterm = cos(center_freq*i);

    dcmplx t1(gauss*cosineterm,gauss*sineterm);
    dcmplx t2(-i/pow(sigma,2),center_freq);
    filter[i+offset] = t1*t2;
  }


  if(PRINT){
    cout << "Derivative kernel along x direction for filter 0: " << endl;

    if(test == 0){
      for (int i=-offset; i<=offset; i++){//frame
	cout << filter[i+offset] <<"\t";
      }

    }

    test = 1;
  }
}

/*************************************************************************/
/* Compute all derivative quantities, tries to use what it can of        */
/* available filtered files                                              */
/*************************************************************************/

void compute_derivatives(double ***img,string filename,string tempPath,string filtPath, int *pic_size,int num_scales,int central_image){

  ostringstream framenum;
  framenum << central_image;

  int found_img_file;
  int filter_size[3],len; 

  dcmplx **filt_output; 
  dcmplx *deri_filter_x,*deri_filter_y,*deri_filter_tt;

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  double **filter_centerfreq;

  string gab_string;

  /* Compute the center frequencies of the Gabor filters */
  initFilterParameters(num_scales,size,radius,sigma);
  filter_centerfreq = init_filters(radius,num_scales);

  filt_output = alloc2dcomplexarray(pic_size[0],pic_size[1]);

  cout << "\nComputing derivatives for filter: ";

  /* Compute the derivatives - Rx,Ry,Rt */
  for(int i=0; i<num_scales; i++){

    ostringstream scalenum;
    scalenum << i;

    cout << "\nScale: " << i << ":\n\n";
    len = size[i];
    filter_size[0] = filter_size[1] = filter_size[2] = len;
    int offset = len/2;

    /* Declare variables to hold the filter */
    deri_filter_x = new dcmplx[len];
    deri_filter_y = new dcmplx[len];
    deri_filter_tt = new dcmplx[len];

    for(int j=0; j<NFILTERS; j++){

      cout << i*NFILTERS+j << "\t" ;
      fflush(stdout);
      int filtnum = i*NFILTERS+j;
      ostringstream filenum;            
      filenum << filtnum;

      for(int k=0; k<3; k++){

	switch(k){
	case 0: 
	  gab_string = "gabx";
	  create_derivative_kernel(deri_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_gabor_1d(deri_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_gabor_1d(deri_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);
	  break;
	case 1:
	  gab_string = "gaby";
	  create_gabor_1d(deri_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_derivative_kernel(deri_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_gabor_1d(deri_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);
	  break;
	case 2:
	  gab_string = "gabt";
	  create_gabor_1d(deri_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_gabor_1d(deri_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_derivative_kernel(deri_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);
	  break;
	} 

	string imgbase = filename + ".scale" + scalenum.str() + ".frame" + framenum.str() + "." + gab_string + filenum.str();	  
	string imgout = filtPath + "/" + imgbase;
	ifstream in;
	in.open(imgout.c_str(),ios::in|ios::binary);

	if(in.fail()){
	  found_img_file = FALSE;
	}
	else{
	  found_img_file = TRUE;
	  in.close();
	}

	if(!found_img_file){
	  separable_convolve(filt_output,img,deri_filter_x,deri_filter_y,deri_filter_tt,pic_size,len,len,len);	 
	  write_filtered_files(tempPath,imgbase,pic_size,offset,filt_output);
	}
      }
    }

    cout << "\n"; 

    /* Deallocate all variables */

    delete [] deri_filter_x;
    delete [] deri_filter_y;
    delete [] deri_filter_tt;
  }



  dealloc2dcomplexarray(filt_output,pic_size[0],pic_size[1]);

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }

  delete [] filter_centerfreq;

  delete [] size;
  delete [] radius;
  delete [] sigma;

}   

/*************************************************************************/
/* Compute all derivative quantities                                     */
/*************************************************************************/

void compute_derivatives(double ***img,string filename,string tempPath,int *pic_size,int num_scales,int central_image){

  ostringstream framenum;
  framenum << central_image;

  dcmplx *deri_filter_x,*deri_filter_y,*deri_filter_tt;
  dcmplx **filt_output;

  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  double **filter_centerfreq;

  /* Compute the center frequencies of the Gabor filters */
  initFilterParameters(num_scales,size,radius,sigma);
  filter_centerfreq = init_filters(radius,num_scales);

  string gab_string;
  int len, filter_size[3]; //[nrows,ncols,nframes]

  /* Declare variables to hold the output */
  filt_output = alloc2dcomplexarray(pic_size[0],pic_size[1]);

  cout << "\nComputing derivatives for filter: ";
  /* Compute the derivatives - Rx,Ry,Rt */
  for(int i=0; i<num_scales; i++){

    ostringstream scalenum;
    scalenum << i;

    len = size[i];
    filter_size[0] = filter_size[1] = filter_size[2] = len;
    int offset = len/2;

    /* Declare variables to hold the filter */
    deri_filter_x = new dcmplx[len];
    deri_filter_y = new dcmplx[len];
    deri_filter_tt = new dcmplx[len];

    for(int j=0; j<NFILTERS; j++){

     
      cout << i*NFILTERS+j << "\t";
      fflush(stdout);

      int filtnum = i*NFILTERS+j;
      ostringstream filenum;            
      filenum << filtnum;

      for(int k=0; k<3; k++){

	switch(k){
	case 0: 
	  gab_string = "gabx";
	  create_derivative_kernel(deri_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_gabor_1d(deri_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_gabor_1d(deri_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);
	  break;
	case 1:
	  gab_string = "gaby";
	  create_gabor_1d(deri_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_derivative_kernel(deri_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_gabor_1d(deri_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);
	  break;
	case 2:
	  gab_string = "gabt";
	  create_gabor_1d(deri_filter_x,filter_centerfreq[i*NFILTERS+j][0],sigma[i],len);
	  create_gabor_1d(deri_filter_y,filter_centerfreq[i*NFILTERS+j][1],sigma[i],len);
	  create_derivative_kernel(deri_filter_tt,filter_centerfreq[i*NFILTERS+j][2],sigma[i],len);
	  break;
	}

	string imgbase = filename + ".scale" + scalenum.str() + ".frame" + framenum.str() + "." + gab_string + filenum.str();	  
	separable_convolve(filt_output,img,deri_filter_x,deri_filter_y,deri_filter_tt,pic_size,len,len,len);
	write_filtered_files(tempPath,imgbase,pic_size,offset,filt_output);
      }
    }

    cout << "\n";

    /* Deallocate all variables */

    delete [] deri_filter_x;
    delete [] deri_filter_y;
    delete [] deri_filter_tt;

  } 

  dealloc2dcomplexarray(filt_output,pic_size[0],pic_size[1]);

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }

  delete [] filter_centerfreq;

  delete [] size;
  delete [] radius;
  delete [] sigma;

}

/*****************************************************************/
/*               Compute MOVIE values                             */
/*****************************************************************/

void compute_movie(string tempPath,string outPath,string refname,string disname,int *pic_size,int num_scales,int central_image,double &smovie, double& tmovie){

  /* Read in all the filter information */
  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  initFilterParameters(num_scales,size,radius,sigma);
  double **filter_centerfreq = init_filters(radius,num_scales);

  /* Set DC parameters */

  double sigma_dc_freq = radius[num_scales-1]-1/sigma[num_scales-1];
  double sigma_dc = 1.0/sigma_dc_freq;
  int size_dc = static_cast<int>(6*sigma_dc + 1);
  if(size_dc%2 == 0){ 
    size_dc += 1;
  }

  /* Read in the flow field */
  double ***full_velocities; //Contains the velocities computed by the Fleet and Jepson algorithm
  full_velocities = alloc3ddoublearray(pic_size[0],pic_size[1],2);
  ostringstream centralFramenum;
  centralFramenum << central_image;
  int offset = size[num_scales-1]/2;
  string flowname = refname + ".frame" + centralFramenum.str() + ".finalflow";
  read_full_velocities(full_velocities,tempPath,flowname);
    
  /* Allocation for MOVIE computation */

  double **movie_temporal = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double ***movie_spatial = alloc3ddoublearray(pic_size[0],pic_size[1],num_scales+1);

  /*   Allocation for spatial variables */

  int response_size[2] = {pic_size[0],pic_size[1]};
  dcmplx **ref_spatial_coeff = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  dcmplx **dis_spatial_coeff = alloc2dcomplexarray(pic_size[0],pic_size[1]);
  double **ref_2 = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **dis_2 = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **rd_prd = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **rd_prd_filtered = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **ref_2_filtered = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **dis_2_filtered = alloc2ddoublearray(pic_size[0],pic_size[1]);

  /* Allocation for temporal variables*/

  double **ref_vel_num = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **dis_vel_num = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **ref_vel_den = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **dis_vel_den = alloc2ddoublearray(pic_size[0],pic_size[1]);

  /* Allocation for dc variables */

  double **ref_local_mean = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **dis_local_mean = alloc2ddoublearray(pic_size[0],pic_size[1]);

  double **ref_dc = alloc2ddoublearray(pic_size[0],pic_size[1]);
  double **dis_dc = alloc2ddoublearray(pic_size[0],pic_size[1]);

  /* Initialize all variables */

  initialize_values(movie_temporal,pic_size[0],pic_size[1],0,0,0);
  initialize_values(movie_spatial,pic_size[0],pic_size[1],num_scales+1,0,0,0);
  initialize_values(ref_vel_num,pic_size[0],pic_size[1],0,0,0);
  initialize_values(dis_vel_num,pic_size[0],pic_size[1],0,0,0);
  initialize_values(ref_vel_den,pic_size[0],pic_size[1],0,0,0);
  initialize_values(dis_vel_den,pic_size[0],pic_size[1],0,0,0);
  initialize_values(ref_2,pic_size[0],pic_size[1],0,0,0);
  initialize_values(dis_2,pic_size[0],pic_size[1],0,0,0);
 
  string fileout;

  /* Check if computed results are available */

  int is_ref_velocity = FALSE;
  int is_dis_velocity = FALSE;
  int *is_spatial = new int[num_scales+1];

  check_temporal(tempPath,refname,disname,central_image,is_ref_velocity,is_dis_velocity,ref_vel_num,ref_vel_den,dis_vel_num,dis_vel_den);
  check_spatial(tempPath,outPath,refname,disname,central_image,num_scales,is_spatial,movie_spatial);
  double *spatial_qual = new double[num_scales+1];
  int is_all_spatial = TRUE;
  for(int m=0; m<num_scales+1;m++){
    is_all_spatial &= is_spatial[m];
  }
  if(is_all_spatial){
    cout << "Spatial MOVIE found\n";
  }

  /* Computing weights for all filters based on the flow vector */

  int total_filters = num_scales*NFILTERS;
  double ***weights = alloc3ddoublearray(pic_size[0],pic_size[1],total_filters+1);
  design_weights(weights,full_velocities,tempPath,refname,pic_size,num_scales,central_image);

  /* Create a Gaussian window for weighted statistic computation */
  
  int size_spmw = SPMOVIE_WINDOW;
  int windowsize = size_spmw*size_spmw;
  double windowsum = 0;  
  double *gauss_window = new double[windowsize];
  double *gaussx = new double[size_spmw];

  create_gauss_1d(gaussx,size_spmw/3,size_spmw);
    
  for(int gx = 0;gx<size_spmw; gx++){
    for(int gy = 0; gy<size_spmw; gy++){
      gauss_window[gx*size_spmw+gy] = gaussx[gx]*gaussx[gy];
      windowsum+=gauss_window[gx*size_spmw+gy];
    }
  }
  
  // Make the windows unit sum
  
  for(int gx = 0;gx<size_spmw; gx++){
    for(int gy = 0; gy<size_spmw; gy++){
      gauss_window[gx*size_spmw+gy] /= windowsum;
    }
  }

  for(int gx = 0;gx<size_spmw; gx++){
    gaussx[gx] /= sqrt(windowsum);
  }

  /* Let's work with the DC component. Compute local mean first. */
    
  if(!is_ref_velocity || !is_dis_velocity || !is_all_spatial){

    double *ref_spa_tuned = new double[windowsize];
    double *dis_spa_tuned = new double[windowsize];

    string refbase = refname+".frame"+centralFramenum.str()+".dc";
    read_velocity_response(tempPath+"/"+refbase,ref_dc,response_size,offset);
    
    string disbase = disname+".frame"+centralFramenum.str()+".dc";
    read_velocity_response(tempPath+"/"+disbase,dis_dc,response_size,offset);

    offset = size[num_scales-1]/2;

    for(int i=offset; i<pic_size[0]-offset; i++){
      for(int j=offset; j<pic_size[1]-offset; j++){
	
	for(int ssi=0; ssi<size_spmw;ssi++){
	  for(int ssj=0; ssj<size_spmw;ssj++){
	    if(!is_ref_velocity || !is_all_spatial){
	      ref_spa_tuned[ssi*size_spmw+ssj] = ref_dc[i+ssi-size_spmw/2][j+ssj-size_spmw/2];
	    }
	    if(!is_dis_velocity || !is_all_spatial){
	      dis_spa_tuned[ssi*size_spmw+ssj] = dis_dc[i+ssi-size_spmw/2][j+ssj-size_spmw/2];
	    }
	  }
	}
	
	/* Computing weighted mean locally */
	  
	if(!is_ref_velocity || !is_all_spatial){
	  ref_local_mean[i][j] = gsl_stats_wmean(gauss_window,1,ref_spa_tuned,1,windowsize);
	  assert(ref_local_mean[i][j] > 0);
	  if(!is_ref_velocity){
	    double dcvar = pow(abs(ref_dc[i][j]-ref_local_mean[i][j]),2.0);
	    ref_vel_num[i][j] += (weights[num_scales*NFILTERS][i][j]*dcvar);
	    ref_vel_den[i][j] += dcvar;
	  }
	}
	if(!is_dis_velocity || !is_all_spatial){
	  dis_local_mean[i][j] = gsl_stats_wmean(gauss_window,1,dis_spa_tuned,1,windowsize);
	  assert(dis_local_mean[i][j]>0);
	  if(!is_dis_velocity){
	    double dcvar = pow(abs(dis_dc[i][j]-dis_local_mean[i][j]),2.0);
	    dis_vel_num[i][j] += (weights[num_scales*NFILTERS][i][j]*dcvar);
	    dis_vel_den[i][j] += dcvar;
	  }
	}
      }
    }

    if(!is_ref_velocity || !is_all_spatial){
      string fileout = tempPath + "/" + refname + ".frame" + centralFramenum.str() + ".means";
      write_velocity_response(fileout,ref_local_mean,response_size,offset);
    }

    if(!is_dis_velocity || !is_all_spatial){
      string fileout = tempPath + "/" + disname + ".frame" + centralFramenum.str() + ".means";
      write_velocity_response(fileout,dis_local_mean,response_size,offset);
    }

    /* DC component of Spatial MOVIE */
 
    offset = size[num_scales-1]/2 + size_spmw/2;

    if(!is_all_spatial){

      for(int i=offset;i<pic_size[0]-offset;i++){
	for(int j=offset;j<pic_size[1]-offset;j++){

	  for (int ssi=0; ssi<SPMOVIE_WINDOW; ssi++){
	    for(int ssj=0; ssj<SPMOVIE_WINDOW; ssj++){
	      ref_2[i][j] += (gauss_window[ssi*SPMOVIE_WINDOW+ssj]*pow(ref_dc[i+ssi-SPMOVIE_WINDOW/2][j+ssj-SPMOVIE_WINDOW/2]-ref_local_mean[i][j],2.0));
	      dis_2[i][j] += (gauss_window[ssi*SPMOVIE_WINDOW+ssj]*pow(dis_dc[i+ssi-SPMOVIE_WINDOW/2][j+ssj-SPMOVIE_WINDOW/2]-dis_local_mean[i][j],2.0));
	    }
	  }
	}
      }

      for(int i=offset;i<pic_size[0]-offset;i++){
	for(int j=offset;j<pic_size[1]-offset;j++){

	  double error = 0;
	  double max_mask = gsl_max(sqrt(ref_2[i][j]),sqrt(dis_2[i][j]))+K_DC;
 
	  for (int ssi=0; ssi<SPMOVIE_WINDOW; ssi++){
	    for(int ssj=0; ssj<SPMOVIE_WINDOW; ssj++){
	      double ref_masked = abs(ref_dc[i+ssi-SPMOVIE_WINDOW/2][j+ssj-SPMOVIE_WINDOW/2]-ref_local_mean[i][j]);
	      double dis_masked = abs(dis_dc[i+ssi-SPMOVIE_WINDOW/2][j+ssj-SPMOVIE_WINDOW/2]-dis_local_mean[i][j]);
	      error += (gauss_window[ssi*SPMOVIE_WINDOW+ssj]*pow(ref_masked/max_mask-dis_masked/max_mask,2.0));
	    }
	  }
	  movie_spatial[num_scales][i][j] = error/2;
	  assert(abs(movie_spatial[num_scales][i][j]) < 1+TOL);
	}
      }

    }
   
    delete [] ref_spa_tuned;
    delete [] dis_spa_tuned;
    
  }

  if(!is_all_spatial){
    ostringstream scalenum;
    scalenum << num_scales;
    fileout = outPath + "/" + disname + ".scale" + scalenum.str() + ".frame" + centralFramenum.str() + ".smoviemap";
    write_movie_maps(fileout,movie_spatial[num_scales],pic_size,offset);
  }

  int is_scales_spatial = TRUE;
  for(int m=0; m<num_scales;m++){
    is_scales_spatial &= is_spatial[m];
  }
    
  /* Spatial and Temporal MOVIE for all other Gabor filters */

  if(!is_dis_velocity || !is_ref_velocity || !is_scales_spatial){
    cout << "\nComputing velocity tuned response... \n";
    for(int m=0; m<num_scales; m++){

      cout << "\nScale: " << m << ": \n\n";

      ostringstream scalenum;
      scalenum << m;

      for(int n=0; n<NFILTERS; n++){

	fflush(stdout);

	/* Read in the response of the reference and distorted images to this filter */

	int filtnum = m*NFILTERS+n;
	cout << m*NFILTERS+n << "\t";
	fflush(stdout);

	/* Convert integer to string */
	ostringstream filenum;
	filenum << filtnum;
	
	for(int frameind = central_image; frameind <= central_image; frameind++){

	  ostringstream framenum;
	  framenum << frameind;
	  
	  if(!is_ref_velocity || !is_scales_spatial){
	    string refbase = refname+".scale"+scalenum.str()+".frame"+framenum.str()+".gab"+filenum.str();
	    read_filtered_files(tempPath,refbase,response_size,offset,ref_spatial_coeff);
	  }

	  if(!is_dis_velocity || !is_scales_spatial){
	    string disbase = disname+".scale"+scalenum.str()+".frame"+framenum.str()+".gab"+filenum.str();
	    read_filtered_files(tempPath,disbase,response_size,offset,dis_spatial_coeff);
	  }
	  
	  offset = size[num_scales-1]/2+size_spmw/2;
  
	  for(int i=offset; i<pic_size[0]-offset; i++){
	    for(int j=offset; j<pic_size[1]-offset; j++){
	      
	      if(!is_ref_velocity){
		assert(ref_local_mean[i][j]>0);
		double c = pow(abs(ref_spatial_coeff[i][j]),2.0);
		ref_vel_num[i][j] += (weights[filtnum][i][j]*c);
		ref_vel_den[i][j] += c;
	      }

	      if(!is_dis_velocity){
		assert(dis_local_mean[i][j]>0);
		double c = pow(abs(dis_spatial_coeff[i][j]),2.0);
		dis_vel_num[i][j] += (weights[filtnum][i][j]*c);
		dis_vel_den[i][j] += c;
	      }

	      if(!is_spatial[m]){

		rd_prd[i][j] = abs(ref_spatial_coeff[i][j])*abs(dis_spatial_coeff[i][j]);
		ref_2[i][j] = pow(abs(ref_spatial_coeff[i][j]),2.0);
		dis_2[i][j] = pow(abs(dis_spatial_coeff[i][j]),2.0);

	      }
	    }
	  }		  
	    
	  if(!is_spatial[m]){
	    separable_convolve(ref_2_filtered,ref_2,gaussx,gaussx,pic_size,size_spmw,size_spmw);
	    separable_convolve(dis_2_filtered,dis_2,gaussx,gaussx,pic_size,size_spmw,size_spmw);
	    separable_convolve(rd_prd_filtered,rd_prd,gaussx,gaussx,pic_size,size_spmw,size_spmw);
	    
	    offset = size[num_scales-1]/2+size_spmw;
	    
	    for(int i=offset; i<pic_size[0]-offset; i++){
	      for(int j=offset; j<pic_size[1]-offset; j++){
		double max_mask = gsl_max(sqrt(ref_2_filtered[i][j]),sqrt(dis_2_filtered[i][j]))+K_SP;
		double t1 = ref_2_filtered[i][j]/pow(max_mask,2.0);
		double t2 = dis_2_filtered[i][j]/pow(max_mask,2.0);
		double t3 = rd_prd_filtered[i][j]/pow(max_mask,2.0);
		movie_spatial[m][i][j] += ((t1+t2-2*t3)/(2*NFILTERS));
		assert(movie_spatial[m][i][j] < 1+TOL);
	      }
	    }
	  }
	}
      }
    }
  }

  dealloc2ddoublearray(ref_2,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(dis_2,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(rd_prd,pic_size[0],pic_size[1]);

  dealloc2ddoublearray(ref_2_filtered,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(dis_2_filtered,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(rd_prd_filtered,pic_size[0],pic_size[1]);

  /* Write Spatial MOVIE maps and velocity tuned responses to file */

  int vel_size[2] = {pic_size[0],pic_size[1]};

  if(!is_dis_velocity){
    string fileout = tempPath + "/" + disname + ".frame" + centralFramenum.str() + ".velNum";
    write_velocity_response(fileout,dis_vel_num,vel_size,offset);
    fileout = tempPath + "/" + disname + ".frame" + centralFramenum.str() + ".velDen";
    write_velocity_response(fileout,dis_vel_den,vel_size,offset);
  }

  if(!is_ref_velocity){
    string fileout = tempPath + "/" + refname + ".frame" + centralFramenum.str() + ".velNum";
    write_velocity_response(fileout,ref_vel_num,vel_size,offset);
    fileout = tempPath + "/" + refname + ".frame" + centralFramenum.str() + ".velDen";
    write_velocity_response(fileout,ref_vel_den,vel_size,offset);
  }

  if(!is_all_spatial){
    offset = size[num_scales-1]/2+2*size_spmw/2;
    for(int m=0; m<num_scales; m++){
      if(!is_spatial[m]){
	ostringstream scalenum;
	scalenum << m;
	fileout = outPath + "/" + disname + ".scale" + scalenum.str() + ".frame" + centralFramenum.str() + ".smoviemap";
	write_movie_maps(fileout,movie_spatial[m],pic_size,offset);
      }
    }
  }

  dealloc2dcomplexarray(ref_spatial_coeff,pic_size[0],pic_size[1]);
  dealloc2dcomplexarray(dis_spatial_coeff,pic_size[0],pic_size[1]);

  /* Compute Spatial MOVIE index per frame using the Coefficient of Variation of the map */

  offset = size[num_scales-1]/2 + size_spmw/2;
  long int framesize = (pic_size[0]-2*offset)*(pic_size[1]-2*offset);
  double *smovie_values = new double[framesize];

  for(int i=offset;i<pic_size[0]-offset;i++){
    for(int j=offset;j<pic_size[1]-offset;j++){
      long int index = (i-offset)*(pic_size[1]-2*offset)+(j-offset);
      smovie_values[index] = 0;
      for(int m=0; m<num_scales+1; m++){
	smovie_values[index] += movie_spatial[m][i][j];
      }
      smovie_values[index]/=4;
    }
  }

  double smovie_mean = gsl_stats_mean(smovie_values,1,(pic_size[0]-2*offset)*(pic_size[1]-2*offset));
  double smovie_std = gsl_stats_sd(smovie_values,1,(pic_size[0]-2*offset)*(pic_size[1]-2*offset));
  smovie = smovie_std/(1-smovie_mean);
  delete [] smovie_values;

  offset = size[num_scales-1]/2+size_spmw+TMOVIE_WINDOW/2;

  for(int i=offset; i<pic_size[0]-offset; i++){
    for(int j=offset; j<pic_size[1]-offset; j++){

      double ref_vel,dis_vel,temporal_qual=0;
      for (int ssi=0;ssi<TMOVIE_WINDOW;ssi++){
	for(int ssj = 0; ssj<TMOVIE_WINDOW;ssj++){

	  ref_vel = (ref_vel_num[i+ssi-TMOVIE_WINDOW/2][j+ssj-TMOVIE_WINDOW/2])/(ref_vel_den[i+ssi-TMOVIE_WINDOW/2][j+ssj-TMOVIE_WINDOW/2]+K_SUM);
	  dis_vel = (dis_vel_num[i+ssi-TMOVIE_WINDOW/2][j+ssj-TMOVIE_WINDOW/2])/(dis_vel_den[i+ssi-TMOVIE_WINDOW/2][j+ssj-TMOVIE_WINDOW/2]+K_SUM);
	  
	  temporal_qual += (gauss_window[ssi*TMOVIE_WINDOW+ssj]*pow(ref_vel-dis_vel,2.0));

	}
      }

      movie_temporal[i][j] = temporal_qual;
      assert(movie_temporal[i][j] <= 2+TOL);
    }
  }

  delete [] spatial_qual;

  fileout = outPath + "/" + disname + ".frame" + centralFramenum.str() + ".tmoviemap";
  write_movie_maps(fileout,movie_temporal,pic_size,offset);

  /* Compute temporal MOVIE index per frame using the Coefficient of Variation of the map */
  
  double *tmovie_values = new double[framesize];

  for(int i=offset;i<pic_size[0]-offset;i++){
    for(int j=offset;j<pic_size[1]-offset;j++){
      long int index = (i-offset)*(pic_size[1]-2*offset)+j-offset;
      tmovie_values[index] = movie_temporal[i][j];
    }
  }

  double tmovie_mean = gsl_stats_mean(tmovie_values,1,(pic_size[0]-2*offset)*(pic_size[1]-2*offset));
  double tmovie_std = gsl_stats_sd(tmovie_values,1,(pic_size[0]-2*offset)*(pic_size[1]-2*offset));
  tmovie = tmovie_std/(1-tmovie_mean);
  delete [] tmovie_values;

  /* Clean up and return */

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }

  delete [] filter_centerfreq;

  delete [] size;
  delete [] sigma;
  delete [] radius;
  delete [] is_spatial;
  delete [] gaussx;
  delete [] gauss_window;

  dealloc2ddoublearray(movie_temporal,pic_size[0],pic_size[1]);
  dealloc3ddoublearray(movie_spatial,pic_size[0],pic_size[1],num_scales+1);
  dealloc2ddoublearray(ref_vel_num,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(dis_vel_num,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(ref_vel_den,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(dis_vel_den,pic_size[0],pic_size[1]);
  dealloc3ddoublearray(full_velocities,pic_size[0],pic_size[1],2);

  dealloc3ddoublearray(weights,pic_size[0],pic_size[1],total_filters+1);
  dealloc2ddoublearray(ref_local_mean,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(dis_local_mean,pic_size[0],pic_size[1]);
  dealloc2ddoublearray(ref_dc,response_size[0],response_size[1]);
  dealloc2ddoublearray(dis_dc,response_size[0],response_size[1]);
}

/***********************************************************************************************/
/*              Check if velocity tuned responses are available                                */
/***********************************************************************************************/

void check_temporal(string tempPath,string refname,string disname, int framenum, int& is_ref_vel, int& is_dis_vel,double **ref_vel_num,double **ref_vel_den,double **dis_vel_num,double **dis_vel_den){

  ostringstream centralFramenum;
  centralFramenum << framenum;  

  string fileout = tempPath + "/" + refname + ".frame" + centralFramenum.str() + ".velNum";
  ifstream inref(fileout.c_str(),ios::in | ios::binary);

  if(IS_VEL && !inref.fail()){
    inref.close();
    cout << "Velocity tuned response for reference video found \n";
    is_ref_vel = TRUE;
    int vel_size[2];
    int offset;
    read_velocity_response(fileout,ref_vel_num,vel_size,offset);
    fileout = tempPath + "/" + refname + ".frame" + centralFramenum.str() + ".velDen";
    read_velocity_response(fileout,ref_vel_den,vel_size,offset);
  }
  else{
    inref.close();
  }

  fileout = tempPath + "/" + disname + ".frame" + centralFramenum.str() + ".velNum";
  ifstream indis(fileout.c_str(),ios::in | ios::binary);

  if(IS_VEL && !indis.fail()){
    indis.close();
    cout << "Velocity tuned response for distorted video found \n";
    is_dis_vel = TRUE;
    int vel_size[2];
    int offset;
    read_velocity_response(fileout,dis_vel_num,vel_size,offset);
    fileout = tempPath + "/" + disname + ".frame" + centralFramenum.str() + ".velDen";
    read_velocity_response(fileout,dis_vel_den,vel_size,offset);
  }
  else{
    indis.close();
  }
}

/***********************************************************************************************/
/*              Check if Spatial MOVIE maps are found                                          */
/***********************************************************************************************/

void check_spatial(string tempPath,string outPath,string refname,string disname,int framenum,int num_scales,int *is_spatial,double ***movie_spatial){

  ostringstream centralFramenum;
  centralFramenum << framenum;  

  int resp_size[2],offset;
  for(int m=0; m<num_scales+1; m++){
    ostringstream scalenum;
    scalenum << m;
    string fileout = outPath + "/" + disname + ".scale" + scalenum.str() + ".frame" + centralFramenum.str() + ".smoviemap";
    ifstream inref(fileout.c_str(),ios::in | ios::binary); 
    if(IS_VEL && !inref.fail()){
      inref.close();
      is_spatial[m] = TRUE;
      read_movie_maps(fileout,movie_spatial[m],resp_size,offset);
    }
    else{
      is_spatial[m] = FALSE;
    }
  }
}

/***************************************************************************************************/
/*                   Design weights                                                                */
/***************************************************************************************************/

void design_weights(double ***weights,double ***full_velocities,string tempPath,string refname,int *pic_size,int num_scales,int central_image){

  /* Read in all the filter information */
  int *size = new int[num_scales];
  double *radius = new double[num_scales];
  double *sigma = new double[num_scales];
  initFilterParameters(num_scales,size,radius,sigma);
  double **filter_centerfreq = init_filters(radius,num_scales);

  ostringstream centralFramenum;
  centralFramenum << central_image;

  int total_filters = num_scales*NFILTERS;
  double ***weights_sum = alloc3ddoublearray(pic_size[0],pic_size[1],num_scales);
  double ***weights_max = alloc3ddoublearray(pic_size[0],pic_size[1],num_scales);
  initialize_values(weights_sum,pic_size[0],pic_size[1],num_scales,0,0,0);
  initialize_values(weights_max,pic_size[0],pic_size[1],num_scales,0,0,0);
  initialize_values(weights,pic_size[0],pic_size[1],total_filters+1,0,0,0);

  int offset = size[num_scales-1]/2;
  for(int m=0; m<num_scales;m++){
    double rad = radius[m];
    for(int n=0;n<NFILTERS; n++){
      int filtnum = m*NFILTERS+n;
      for(int i=offset; i<pic_size[0]-offset; i++){
	for(int j=offset; j<pic_size[1]-offset; j++){
	  double vx=0;
	  double vy = 0;

	  if(m==num_scales-1 && n==NFILTERS-1){ // Account for DC
	    weights[num_scales*NFILTERS][i][j] = 1;
	  }

	  if(full_velocities[1][i][j]!=UNDEFINED && full_velocities[0][i][j]!=UNDEFINED){
	    vx = full_velocities[1][i][j];
	    vy = full_velocities[0][i][j];
	  }
	  
	  double dist = compute_dist_from_plane(vx,vy,filter_centerfreq[filtnum]);
	  weights[filtnum][i][j] = (rad-dist)/rad;
	  
	  if((rad-dist)<0){
	    cout << "I found a weight that is negative \n";
	    cout << "Quitting \n";
	    exit(-1);
	  }
	  weights_sum[m][i][j] += weights[filtnum][i][j];
	  if(weights[filtnum][i][j]>weights_max[m][i][j]){
	    weights_max[m][i][j] = weights[filtnum][i][j];
	  }
	}
      }
    }
  }
    
  /* Converting weights to zero mean and unit maximum */

  for(int i=offset; i<pic_size[0]-offset; i++){
    for(int j=offset; j<pic_size[1]-offset; j++){
      
      for(int m=0; m<num_scales;m++){
	for(int n=0;n<NFILTERS; n++){
	  double wmean = (weights_sum[m][i][j]/NFILTERS);
	  weights[m*NFILTERS+n][i][j] -= wmean;
	  weights[m*NFILTERS+n][i][j] /= (weights_max[m][i][j]-wmean);
	}
      }
    }
  }

  for(int i=0; i<NFILTERS*num_scales; i++){
    delete [] filter_centerfreq[i];
  }

  delete [] filter_centerfreq; 

  delete [] size;
  delete [] sigma;
  delete [] radius;
  dealloc3ddoublearray(weights_sum,pic_size[0],pic_size[1],num_scales);
  dealloc3ddoublearray(weights_max,pic_size[0],pic_size[1],num_scales);

}  

