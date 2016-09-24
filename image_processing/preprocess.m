function [ data ] = preprocess( movie_in, movie_out, varargin)
    % high pass filters, then dfof's, and masks for
    % vasculature. 
    %
    % Usage M = preprocess(Min, Mout, 'sigma', 40, 'verbose', 1);
    % Min and Mout are filenames.
    % Optional inputs given by 'sigma' and 'verbose'
    %
    % To do: separate these into util functions, and add options for
    % everything. 
    %
    % Change log:
    % 2016.09.17 - Amy JC - Created
    % 2016.09.23 - Shai - Added some optional inputs for verbosity and the
    % high pass filter size
    
    
    
    %% Process optional inputs
    p = inputParser;
    defaultSigmaValue = 40;
    defaultVerbose = 0;
    addParameter(p,'sigma',defaultSigmaValue,@isnumeric);
    addParameter(p,'verbose',defaultVerbose,@isnumeric);
    parse(p,varargin{:});
    sigmaValue = p.Results.sigma;
    verbose = p.Results.verbose;
   
    
    %% per frame normalization   
    data = load_movie(movie_in);
    
    if(verbose == 1)
        figure(100); subplot(5,1,1); title('loaded movie');
        f = compute_fluorescence_stats(data); plot(f);
    end
    
    av = mean(mean(data));
    data = bsxfun(@rdivide, data, av);
    
    if(verbose == 1)
        figure(100); subplot(5,1,2); title('per frame normalization');
        f = compute_fluorescence_stats(data); plot(f);
    end 
   
    
    %% high pass filter
    %sigmaValue = 40;
    FilterSize=round(6*sigmaValue);
    AppliedFilter=fspecial('gaussian',FilterSize,sigmaValue);
    if FilterSize/2==round(FilterSize/2)
        Identity=padarray([0.25 0.25;0.25 0.25],[(FilterSize)/2-1 (FilterSize)/2-1]);
    else
        Identity=padarray(1,[(FilterSize-1)/2 (FilterSize-1)/2]);
    end
    AppliedFilter=Identity-AppliedFilter;

    %then we apply the filter
    mean_image = mean(data, 3);
    h = waitbar(0, 'filtering the image');
    for i = 1: size(data, 3)
        waitbar(i/size(data, 3), h);
        data(:,:,i) = imfilter(data(:,:, i) - mean_image, AppliedFilter) + mean_image;
    end
    close(h);
    
     if(verbose == 1)
        figure(100); subplot(5,1,3); title('high pass filter');
        f = compute_fluorescence_stats(data); plot(f);
    end
    
    mask = imfilter(mean_image, AppliedFilter);
    
    %% DFOF
    disp('calculating dfof...')
    mean_i = mean(data, 3);
    data = bsxfun(@minus, data, mean_i);
    data = bsxfun(@rdivide, data, mean_i);
    
    if(verbose == 1)
        figure(100); subplot(5,1,4); title('high pass filter');
        f = compute_fluorescence_stats(data); plot(f);
    end

    %% Vasculature Mask
    disp('making vasculature mask...')
    mask = imerode(mask, strel('disk', 2));
    mask = medfilt2(mask, [10 10]);
    
    %get user input to define vasculature mask
    %to-do: make this an option. 
    
    ax = imtool(mask, []);
    disp('after selecting threshold press enter to continue...')
    pause;
    
    mask = getimage(ax);
    
    disp('got the mask, applying to movie...')
    h = waitbar(0, 'masking the movie');
    
   %Now we mask! replace anywhere the mask was a zero with the mean dfof
   %image. 
    
    mask_av = mean(data, 3);
    idx = find(mask == 0);
    for i = 1:size(data, 3)
        waitbar(i/size(data, 3), h);
        im = data(:,:,i);
        im(idx) = mask_av(idx);
        data(:,:, i) = im;
    end;
    close(h);
    
   if(verbose == 1)
        figure(100); subplot(5,1,5); title('masked');
        f = compute_fluorescence_stats(data); plot(f);
   end
    
   %% Save output movie
    save_movie_to_hdf5(data, movie_out);
    
end

