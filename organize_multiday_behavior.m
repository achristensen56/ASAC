function output_traces = organize_multiday_behavior(md,ob_list)
    
    function x = subtract_min(in)        
       x = in - min(in); 
    end

    output_traces = [];
    for i = 1:size(md.matched_indices,1)
        traces = cell(1, md.num_days);
        for j = 1:md.num_days
            traces{j} = ob_list(j).traces(:,i)';        
        end
        
        % z-score each day individually
        traces = cellfun(@zscore,traces,'UniformOutput',false);

        % min substract
        traces = cellfun(@subtract_min,traces,'UniformOutput',false);

        full_trace = cell2mat(traces);
        output_traces = [output_traces; full_trace];

    end

    output_traces = output_traces';
    
    
    
    
    
    temp_obj = ob_list(1);
    frame_offset = temp_obj.total_frames;
    
    for i =2:size(ob_list,2)
        temp_obj.stim_right = [temp_obj.stim_right ob_list(i).stim_right];
        temp_obj.response_right = [temp_obj.response_right ob_list(i).response_right];
        temp_obj.prior_right = [temp_obj.prior_right ob_list(i).prior_right];
        temp_obj.coherence = [temp_obj.coherence ob_list(i).coherence];
        temp_obj.was_correct = [temp_obj.was_correct ob_list(i).was_correct];
        temp_obj.num_pokes = [temp_obj.num_pokes ob_list(i).num_pokes];
        temp_obj.total_frames = temp_obj.total_frames + ob_list(i).total_frames;
        temp_obj.trial_completed = [temp_obj.trial_completed ob_list(i).trial_completed];
        
        temp_obj.start_poke_frame = [temp_obj.start_poke_frame frame_offset+ob_list(i).start_poke_frame];
        temp_obj.stim_start_frame = [temp_obj.stim_start_frame frame_offset+ob_list(i).stim_start_frame];
        temp_obj.end_poke_frame = [temp_obj.end_poke_frame frame_offset+ob_list(i).end_poke_frame];
        temp_obj.reward_frame = [temp_obj.reward_frame frame_offset+ob_list(i).reward_frame];
        temp_obj.stim_end_frame = [temp_obj.stim_end_frame frame_offset+ob_list(i).stim_end_frame];
        
        
        frame_offset = frame_offset + ob_list(i).total_frames;
    end
    
    % if final trial was not finished fix lengths
    if length(temp_obj.stim_right)>length(temp_obj.stim_end_frame)
        temp_obj.reward_frame(length(temp_obj.stim_right)) = 0;
        temp_obj.stim_end_frame(length(temp_obj.stim_right)) = 0;
        temp_obj.trial_completed(length(temp_obj.stim_right)) = 0;
    end


    temp_obj = struct(temp_obj);
    
    
    save(['multiday_data.pyd'],'temp_obj','-v6')

    
    
    



end