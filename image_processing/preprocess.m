function [ data ] = preprocess( movie_in, movie_out,sigmaValue)
    %high pass filters, then dfof's, and masks for
    %vasculature. Usage M = preprocess(M, 40), where the second argument is
    %the radius of the gaussian filter to apply. 
    %To do: separate these into util functions, and add options for
    %everything. 
    %Amy JC 9/17/16

    %per frame normalization
    
    data = load_movie(movie_in);
    
    av = mean(mean(data));
    data = bsxfun(@rdivide, data, av);
   
    
    %high pass filter
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
    mask = imfilter(mean_image, AppliedFilter);
    
    %then we dfof
    disp('calculating dfof...')
    mean_i = mean(data, 3);
    data = bsxfun(@minus, data, mean_i);
    data = bsxfun(@rdivide, data, mean_i);

    %making vasculature mask
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
    
    save_movie_to_hdf5(data, movie_out);
    
end

