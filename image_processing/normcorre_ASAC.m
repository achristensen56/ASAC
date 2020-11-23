function normcorre_ASAC(name)

    %% read data and convert to double
    % name = 'movie.h5';
    frame = read_file(name,1,1);
    [d1,d2] = size(frame);

    %% perform some sort of deblurring/high pass filtering
    % The function does not load the whole file in memory. Instead it loads 
    % chunks of the file and then saves the high pass filtered version in a 
    % h5 file.

    gSig = 5; 
    gSiz = 3*gSig; 
    psf = fspecial('gaussian', round(2*gSiz), gSig); % param 1 is size of filter and 2 is sigma of filter
    ind_nonzero = (psf(:)>=max(psf(:,1)));
    psf = psf-mean(psf(ind_nonzero));
    psf(~ind_nonzero) = 0;   % only use pixels within the center disk

    [filepath,file_name,ext] = fileparts(name);
    h5_name = fullfile(filepath,[file_name,'_filtered_data.h5']);
    chunksize = 5000;    % read 500 frames at a time
    cnt = 1;

    Yf = single(read_file(name,1,2));
    Y = imfilter(Yf,psf,'symmetric');
    imagesc(Y(:,:,2));
    while (1)  % read filter and save file in chunks
        try
            Yf = single(read_file(name,cnt,chunksize));
        catch
            'file ended';
            break;
        end
        if isempty(Yf)
            break
        else
            tic;
            Y = imfilter(Yf,psf,'symmetric');
            saveash5(Y,h5_name);
            cnt = cnt + size(Y,ndims(Y));
        end
        dum_var = toc;
        disp([num2str(cnt), ' took ', num2str(dum_var), ' seconds '])
    end

    %% first try out rigid motion correction
    % exclude boundaries due to high pass filtering effects
    options_r = NoRMCorreSetParms('d1',d1,'d2',d2,'bin_width',200,...
        'max_shift',40,'iter',1,'correct_bidir',false,'output_type','h5'...
        ,'mem_batch_size',750,'use_parallel',true,'plot_flag',false,...
        'h5_filename', 'mc.h5', 'fr', 40);

    %% register using the high pass filtered data and apply shifts to original data
    tic; [M1,shifts1,template1] = normcorre_batch_ASAC(h5_name,options_r); toc % register filtered data
    % exclude boundaries due to high pass filtering effects
    
    tic; Mr = apply_shifts_normcorre(name,shifts1,options_r); toc % apply shifts to full dataset
    
    %% delete helper files
    delete('movie_filtered_data.h5');
