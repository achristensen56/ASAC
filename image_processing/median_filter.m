function [ M_filt ] = median_filter( M )
%applies a 3x3 median filter to movie to fix hot pixels


    M_filt = zeros(size(M), 'uint16');

    h = waitbar(0, 'filtering the image');
    for i = 1:size(M, 3)
        waitbar(i/size(M, 3), h);

        %apply a 3x3 median filter to each frame to fix hot pixels.
        M_filt(:,:,i) = medfilt2(M(:, :, i));
    end
end

