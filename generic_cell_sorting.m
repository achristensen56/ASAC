M = load_movie_from_hdf5('m_dff_crp.hdf5');

ds = DaySummary('', '');

%ds = DaySummary('','');
%pick which are good cells
classify_cells(ds, M);


%%
%num_classified cells tells you how many cells you have that are good
num_classified_cells = ds.num_classified_cells;

%this gives you a matrix of traces, N_cells x N_movie frames
traces = zeros(num_classified_cells, ds.full_num_frames);
idx = 1;
for k = 1:ds.num_cells
    if ds.is_cell(k)
        traces(idx,:) = ds.get_trace(k);
        idx = idx + 1;
    end
end
%%
%display raster
imagesc(traces)
%%
%save extracted traces
save('traces.mat', 'traces')

% ds = DaySummary('', '');
% decision = zeros(1,ds.num_cells);
% for k = 1:ds.num_cells % go through every possible cell
%     this_trace = ds.get_trace(k);
%     rho = corr(this_trace',traces');
%     if max(rho)>0.99
%         decision(k) = 1;
%     end      
% end
% 
% save('decision.mat','decision')

masks = [];
for k = 1:ds.num_cells
    if ds.is_cell(k)
        masks = cat(3,masks,ds.get_mask(k));
    end
end

save('masks.mat','masks')




%%
%this gives you a matrix of traces, N_cells x N_movie frames
load('classifierOutputChoices.mat');
num_classified_cells = sum(validCellMax);
traces = zeros(num_classified_cells, ds.full_num_frames);
idx = 1;
for k = 1:ds.num_cells
    if validCellMax(k)
        traces(idx,:) = ds.get_trace(k);
        idx = idx + 1;
        ds.cells(k).label = 'cell';
    else
        ds.cells(k).label = 'not a cell';
    end
end
%%
%display raster
imagesc(traces)
%%
%save extracted traces
save('traces.mat', 'traces')
%save('rec_extract.mat','ds')



