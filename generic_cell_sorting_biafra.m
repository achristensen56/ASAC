% CRITICAL USER PARAMETERS
% Input images and signals, change from PCA-ICA to whatever is appropriate for input from user's cell extraction algorithm.
load('rec_extract.mat')
inputImages = filters; % cell array of [x y nSignals] matrices containing each set of images corresponding to inputSignals objects.
inputSignals = traces'; % cell array of [nSignals frames] matrices containing each set of inputImages signals.
iopts.inputMovie = ['.' filesep 'm_dff.hdf5'];
iopts.inputDatasetName = '/Data/Images'; % HDF5 dataset name

%iopts.inputMovie = ['.' filesep 'mc_ds2_norm_dff.hdf5'];
%iopts.inputDatasetName = '/mov'; % HDF5 dataset name

% MAIN USER parameters: change these as needed
iopts.preComputeImageCutMovies = 0; % Binary: 0 recommended. 1 = pre-compute movies aligned to signal transients, 0 = do not pre-compute.
iopts.readMovieChunks = 1; % Binary: 1 recommended. 1 = read movie from HDD, 0 = load entire movie into RAM.
iopts.showImageCorrWithCharInputMovie = 0; % Binary: 0 recommended. 1 = show the image correlation value when input path to options.inputMovie (e.g. when not loading entire movie into RAM).
iopts.maxSignalsToShow = 9; %Int: max movie cut images to show
iopts.nSignalsLoadAsync = 30; % Int: number of signals ahead of current to asynchronously load imageCutMovies, might make the first couple signal selections slow while loading takes place
iopts.threshold = 0.3; % threshold for thresholding images
iopts.thresholdOutline = 0.3; % threshold for thresholding images

% OPTIONAL
iopts.valid = 'neutralStart'; % all choices start out gray or neutral to not bias user
%iopts.valid = choices;
iopts.cropSizeLength = 20; % region, in px, around a signal source for transient cut movies (subplot 2)
iopts.cropSize = 20; % see above
iopts.medianFilterTrace = 0; % whether to subtract a rolling median from trace
iopts.subtractMean = 0; % whether to subtract the trace mean
iopts.movieMin = -0.01; % helps set contrast for subplot 2, preset movie min here or it is calculated
iopts.movieMax = 0.05; % helps set contrast for subplot 2, preset movie max here or it is calculated
iopts.backgroundGood = [208,229,180]/255;
iopts.backgroundBad = [244,166,166]/255;
iopts.backgroundNeutral = repmat(230,[1 3])/255;

[~, ~, choices] = signalSorter(inputImages, inputSignals, 'options',iopts);
save('choices.mat','choices');
save('filters.mat','filters');
num_classified_cells = sum(choices);

%this gives you a matrix of traces, N_cells x N_movie frames
traces_= zeros(num_classified_cells, size(traces,1));
idx = 1;
for k = 1:length(choices)
    if choices(k)
        traces_(idx,:) = traces(:,k);
        idx = idx + 1;
    end
end

%%
%display raster
imagesc(traces_)

%%
%save extracted traces
save('traces.mat', 'traces_')
save('unsorted_traces.mat','traces')


filt1= zeros(size(filters,1),size(filters,2),num_classified_cells);
idx = 1;
for k = 1:length(choices)
    if choices(k)
        filt1(:,:,idx) = filters(:,:,k);
        idx = idx + 1;
    end
end


filters = filt1;
