function [ m_dfof ] = preprocess( data )
    %high pass filters, then dfof's, and masks for
    %vasculature.
    % To do: separate these into util functions, and add options for
    % everything. 
    % Amy JC 9/17/16

    %per frame normalization
    av = mean(mean(data));
    data = bsxfun(@rdivide, data, av);

    %high pass filter
    sigmaValue = 40;
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
        f_data(:,:,i) = imfilter(data(:,:, i) - mean_image, AppliedFilter) + mean_image;
    end

    %then we dfof
    mean_i = mean(f_data, 3);

    dfof = bsxfun(@minus, f_data, mean_i);

    dfof = bsxfun(@rdivide, dfof, mean_i);

    %then we mask MAKE THE MASK
    img = mean(data, 3);

    img = imfilter(img, AppliedFilter);

    mask1 = imerode(img, strel('disk', 2));
    mask2 = medfilt2(mask1, [10 10]);

    imtool(mask2, []);
    %save output as mask4

    %Now we mask!
    mask_av = mean(dfof, 3);
    idx = find(mask4 == 0);
    for i = 1:size(data, 3)
        im = dfof(:,:,i);
        im(idx) = mask_av(idx);
        m_dfof(:,:, i) = im;
    end;
end

