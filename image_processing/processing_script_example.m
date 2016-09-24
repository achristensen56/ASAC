%% load files, downsample, and median filter
data = load_tif_from_directory('rec*','downsample','preprocess');
save_movie_to_hdf5('movie_nhp_ds.h5');

f = compute_fluorescence_stats(data);
figure(99);subplot(5,1,1);
plot(f);ylabel('loaded movie');

reg_frame = data(:,:,100);
clear data;

%% motion registration

register_movie('movie_nhp_ds.h5','movie_nhp_ds_reg.h5','ref',reg_frame);
data = load_movie_from_hdf5('movie_nhp_ds_reg.h5');

f = compute_fluorescence_stats(data);
figure(99);subplot(5,1,2);
plot(f);ylabel('motion registration');


%% crop
data = data(50:490,60:end,:);

f = compute_fluorescence_stats(data); 
figure(99);subplot(5,1,3);
plot(f);ylabel('cropped');

%% delete first frame
data(:,:,1) =[];

f = compute_fluorescence_stats(data);
figure(99);subplot(5,1,4);
plot(f);ylabel('no first frame');

%% temporal downsample
try
    data = (data(:,:,1:2:end)+data(:,:,2:2:end))/2;
catch
    data = (data(:,:,1:2:end-1)+data(:,:,2:2:end))/2;
end

f = compute_fluorescence_stats(data); 
figure(99);subplot(5,1,5);
plot(f);ylabel('temporal downsample');
save_movie_to_hdf5(data,'movie_nhp_ds_reg_tds.h5');

%% take away another frame
data(:,:,1) =[];
save_movie_to_hdf5(data,'movie_nhp_ds_reg_tds.h5');

%% processing
data = preprocess('movie_nhp_ds_reg_tds.h5','movie_proc.h5','verbose',1,'sigma',40);

%% CELLMAX
movieFilename = '/home/adam/Documents/DATA/Visual_RAM/AC_C1/Day5/CalciumData/CalciumData/DecompressedMovies/movie_proc.h5'
options.movieDatasetName='/Data/Images';
options.eventOptions.framerate = 10;
[output, DFOF] = CELLMax_Wrapper(movieFilename, 'options', options);
save('cellmax_output','output','-v7.3')

%% Sorting cells
import_cellmax(output);
ds = DaySummary([],'.');



