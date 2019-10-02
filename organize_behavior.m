function temp_obj = organize_behavior(fname,trace_file)
    
    load(fname);
    num_trials = obj.curr_trial-1;
    temp_obj.stim_right = obj.behavior_params.correct_side(1:num_trials);


    [temp_obj.start_poke_frame, temp_obj.stim_start_frame, ...
     temp_obj.end_poke_frame,   temp_obj.reward_frame,...
     temp_obj.stim_end_frame, temp_obj.trial_completed] =...
                                                get_frame_data(load(fname),10);

    plot(temp_obj.start_poke_frame);hold on;
    plot(temp_obj.stim_start_frame)
    plot(temp_obj.end_poke_frame)
    plot(temp_obj.reward_frame)
    plot(temp_obj.stim_end_frame)

    %temp_obj.num_trials = num_trials;
    temp_obj.response_right = obj.response.stim_response.response_side;
    temp_obj.prior_right = obj.prob_params.close_priors(1:num_trials);
    %temp_obj.block_length = obj.prob_params.block_length;
    temp_obj.coherence = obj.prob_params.coherence(1:num_trials);

    temp_obj.stim_right(temp_obj.stim_right==3)=0;
    temp_obj.response_right(temp_obj.response_right==3)=0;
    temp_obj.coherence(temp_obj.stim_right==0)=-temp_obj.coherence(temp_obj.stim_right==0);

    temp_obj.response_right(num_trials)=0;
    temp_obj.was_correct = (temp_obj.stim_right==temp_obj.response_right);

    temp_obj.num_pokes = cellfun(@length,obj.response.trial_initiation.start_poke_frame);
    temp_traces = load(trace_file);
    temp_obj.total_frames = size(temp_traces.traces,1);
    temp_obj.traces = temp_traces.traces;
    temp_obj = struct(temp_obj);
   
end



