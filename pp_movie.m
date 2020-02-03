%% preprocessing
function [] = pp_movie(directory)
    cd(directory);
    movie_file = ls('*.isxd');
    convertInscopixIsxdToHdf5(movie_file);
    
    M = load_movie_from_hdf5('movie.h5');
    
    S = sum(M,[1,2]);
    dropped_inds = find(S==0);
    if length(dropped_inds)>0
        for i = 1:length(dropped_inds)
            M(:,:,dropped_inds(i)) = M(:,:,dropped_inds(i)-1);
        end
        delete('movie.h5')
        save_movie_to_hdf5(M,'movie.h5');
    end
    clear all
    
    
    
    
    
    
    
    register_movie('movie.h5','m_reg.hdf5');
% 
%     %view_movie(M);
% 
%     f = compute_fluorescence_stats(M);
%     plot(f)

   %%
%     plot(f(:,3)); %hold on; plot([1 10000],[800 800]);
%     [~,y]=ginput(1);
%     frames_to_replace = find(f(:,3)<y)
%     close all
%     
% for i = 1:length(frames_to_replace)
%     try
%     M(:,:,frames_to_replace(i)) = M(:,:,frames_to_replace(i)-1);
%     catch
%     M(:,:,frames_to_replace(i)) = M(:,:,frames_to_replace(i)+1);
%         
%     end
%     
% end
% 
% save_movie_to_hdf5(M,'m_reg_fixed.hdf5');
%crop_movie('m_reg_fixed.hdf5','m_reg_crp_fixed.hdf5')


% REPLACE DROPPED FRAMES (WILL BE FILLED WITH ZEROS)


% CROP MOVIE
% clear all
% crop_movie('m_reg.hdf5','m_reg_crp_fixed.hdf5','automin')

%%
clear M
M = load_movie_from_hdf5('m_reg.hdf5');
config = [];

config.avg_cell_radius = 7;
config.cellfind_min_snr = 3;
%config.preprocess = false;  % Assuming the movie is already preprocessed
config.dendrite_aware = false;
config.verbose = 2;
config.init_maxnum_iters = 500;
config.downsample_time_by = 1;
config.num_partitions_y = 1;
config.num_partitions_x = 1;
config.downsample_space_by = 1;
config.fix_zero_FOV_strips = true;

%if you get a GPU, then you can use this
config.use_gpu = 1;
config.compute_device = 'gpu';

config = get_defaults(config);
M  = preprocess_movie(M, config);
save_movie_to_hdf5(M,'m_dff.hdf5');


config.preprocess = false;
output = extractor(M, config);


% %collecting the filters and traces from extract
% [d1, d2, d3] = size(output.spatial_weights);
% F1 = reshape(output.spatial_weights,d1*d2, d3);
% T1 = output.temporal_weights';

%plotting the cell map
%

%This block saves the outputs of Extract and runs the classifier
%q = quite, s = save, c = correct, n = not correct
filters = output.spatial_weights;
traces = output.temporal_weights;
info.num_pairs = size(traces, 2);
info.type = 'sana ne amk';
save('rec_extract', 'filters', 'traces', 'info');


% %%
% divisive_normalization_ASAC('m_reg_crp_fixed.hdf5','m_norm.hdf5',10);
% 
% %norm_movie('m_reg_crp_fixed.hdf5','m_norm2.hdf5',20);
% M = load_movie_from_hdf5('m_norm.hdf5');
% m = mean(M, 3);
% M = bsxfun(@minus, M, m);
% M = bsxfun(@rdivide, M, max(m, 1e-6));
% save_movie_to_hdf5(M,'m_dff.hdf5');
% clear M
% %crop_movie('m_dff.hdf5','m_dff_crp.hdf5')\
% 
% delete('m_norm.hdf5');
% delete('m_reg.hdf5');
%  delete('m_reg_fixed.hdf5');
% %delete('m_reg_crp_fixed.hdf5');
% 
% M = ['m_reg_crp_fixed.hdf5', '::', '/Data/Images'];
% M = ['m_reg_crp_fixed.hdf5', '::', '/1'];
% 
% config = [];
% 
% config.avg_cell_radius = 7;
% config.cellfind_min_snr = 3;
% %config.preprocess = false;  % Assuming the movie is already preprocessed
% config.dendrite_aware = false;
% config.verbose = 2;
% config.init_maxnum_iters = 500;
% config.downsample_time_by = 1;
% config.num_partitions_y = 1;
% config.num_partitions_x = 1;
% config.downsample_space_by = 1;
% 
% %if you get a GPU, then you can use this
% config.use_gpu = 1;
% config.compute_device = 'gpu';
% 
% %actually run extract
% output = extractor(M, config);
% 
% 
% % %collecting the filters and traces from extract
% % [d1, d2, d3] = size(output.spatial_weights);
% % F1 = reshape(output.spatial_weights,d1*d2, d3);
% % T1 = output.temporal_weights';
% 
% %plotting the cell map
% %
% 
% %This block saves the outputs of Extract and runs the classifier
% %q = quite, s = save, c = correct, n = not correct
% filters = output.spatial_weights;
% traces = output.temporal_weights;
% info.num_pairs = size(traces, 2);
% info.type = 'sana ne amk';
% save('rec_extract', 'filters', 'traces', 'info');
% 
% 
% 
% %can rereun starting from here (assuming the movie is loaded and named M,
% %will restart classification where you left off
% %M = load_movie('smoothed.hdf5');
%%
% M = load_movie_from_hdf5('m_dff.hdf5');
% 
% %ds = DaySummary('trialdata.txt', '');
% 
% ds = DaySummary('','');
% %pick which are good cells
% classify_cells(ds, M);
% 
% 
% %%
% %num_classified cells tells you how many cells you have that are good
% num_classified_cells = ds.num_classified_cells;
% 
% %this gives you a matrix of traces, N_cells x N_movie frames
% traces = zeros(24, ds.full_num_frames);
% idx = 1;
% for k = 1:ds.num_cells
%      if ds.is_cell(k)
%          traces(idx,:) = ds.get_trace(k);
%          idx = idx + 1;
%      end
%      
% end
% % %%
%  %display raster
%  imagesc(traces)
% %%
% %save extracted traces
% save('traces.mat', 'traces')

cd ..
toc
end