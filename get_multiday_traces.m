function output_traces = get_multiday_traces(md)

    function x = subtract_min(in)        
       x = in - min(in); 
    end

        % initialize output matrix, # neurons by # frames
        output_traces = [];
     
        traces = cell(1, md.num_days);
        for n = 1:md.num_cells
            common_cell_idx = n;
            
            for k = 1:md.num_days

                    day = md.valid_days(k);
                    cell_idx_k = md.get_cell_idx(common_cell_idx, day);
                    ds = md.day(day); % Just a shorthand

                    % Get trace
                    traces{k} = ds.get_trace(cell_idx_k);
            end
             
            
            % z-score each day individually
            traces = cellfun(@zscore,traces,'UniformOutput',false);
            
            % min substract
            traces = cellfun(@subtract_min,traces,'UniformOutput',false);
            
            full_trace = cell2mat(traces);
            output_traces = [output_traces; full_trace];
        end % draw_cell
end
