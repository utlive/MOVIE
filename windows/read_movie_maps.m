function [movie_map,offset] = read_movie_maps(filename)

% Part of the MOVIE Software by Kalpana Seshadrinathan
% (kalpana.seshadrinathan@ieee.org)

fid = fopen(filename,'rb');

if (fid <= 0)
    disp(['Attempting to open file: ',inputfile]);
    disp('Error opening input file \n');
    return;
end;

% First 3 float values indicate the size of input image and the offset. The
% offset represents the location of the first "valid" pixel from MOVIE and
% allows for border processing.

imgSize(1) = fread(fid,1,'int');
imgSize(2) = fread(fid,1,'int');

offset = fread(fid,1,'int');

% Read in the ssim values

movie_map = zeros(imgSize(1),imgSize(2));

t = fread(fid,[imgSize(2),imgSize(1)],'double');
movie_map = t';

fclose(fid);

figure; imagesc(movie_map(offset+1:end-offset,offset+1:end-offset)); axis off; title('MOVIE Map');
