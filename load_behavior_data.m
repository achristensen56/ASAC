function temp_obj = load_behavior_data()

fname = ls('*session1.mat');

load(fname);
num_trials = obj.curr_trial-1;
temp_obj.stim_right = obj.behavior_params.correct_side(1:num_trials);


temp_obj.response_right = obj.response.stim_response.response_side;
temp_obj.prior_right = obj.prob_params.close_priors(1:num_trials);
temp_obj.coherence = obj.prob_params.coherence(1:num_trials);

temp_obj.stim_right(temp_obj.stim_right==3)=0;
temp_obj.response_right(temp_obj.response_right==3)=0;
temp_obj.coherence(temp_obj.stim_right==0)=-temp_obj.coherence(temp_obj.stim_right==0);

temp_obj.response_right(num_trials)=0;
temp_obj.was_correct = (temp_obj.stim_right==temp_obj.response_right);
temp_obj.dots_direction = obj.dots.direction(1:num_trials);

% CHANGE IF GRATINGS/DOTS
temp_obj.dots_direction = obj.dots.gratings.direction(1:num_trials);


temp_obj.num_pokes = cellfun(@length,obj.response.trial_initiation.start_poke_frame);
temp_obj.trial_completed = obj.is_trial_completed(1:num_trials);



temp_obj = struct(temp_obj);
save([fname '.pyd'],'temp_obj','-v6')
fprintf('%s \n', fname);
end