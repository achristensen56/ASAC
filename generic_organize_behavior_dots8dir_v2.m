gpio_file = ls('*.csv');
[start_trial_frame,start_stim_frame, response_frame] = get_gpio_frame_data(gpio_file);

load(ls('*session1.mat'))

all_data = struct;

% dots 8 directions
n_trials = 1:(obj.curr_trial-1);
all_data.was_completed = obj.is_trial_completed(n_trials);
all_data.was_correct = obj.response.stim_response.response_correct(n_trials);
all_data.stim_dir = obj.dots.direction(n_trials);
all_data.correct_side = obj.behavior_params.correct_side(n_trials);
all_data.response_side = obj.response.stim_response.response_side(n_trials);
all_data.traces = load('traces.mat'); all_data.traces = all_data.traces.traces;
all_data.task_info = obj.dots;
frame_info = struct;
frame_info.start_poke_frame = start_trial_frame(n_trials);
frame_info.stim_start_frame = frame_info.start_poke_frame + obj.behavior_params.minCenterTime*10;
frame_info.stim_end_frame =  frame_info.stim_start_frame + obj.behavior_params.min_time_vis*10;
frame_info.response_frame = response_frame(n_trials);
all_data.frame_info = frame_info;


completed_trials_data = struct;



% MAKE STIM AND RESPONSE ALIGNED NEURAL DATA
traces = all_data.traces;
sf = frame_info.stim_start_frame;
rf = frame_info.response_frame;
traces_stim_aligned = [];
traces_resp_aligned = [];
for i = n_trials
    if all_data.was_completed(i)
        traces_stim_aligned = cat(3,traces_stim_aligned,traces(:,sf(i)-40:sf(i)+79));
        traces_resp_aligned = cat(3,traces_resp_aligned,traces(:,rf(i)-60:rf(i)+59));
    end
    
end
traces_stim_aligned = permute(traces_stim_aligned, [3 1 2]);
traces_resp_aligned = permute(traces_resp_aligned, [3 1 2]);

ci = find(all_data.was_completed);
completed_trials_data.was_completed = all_data.was_completed(ci);
completed_trials_data.was_correct = all_data.was_correct(ci);
completed_trials_data.stim_dir = all_data.stim_dir(ci);
completed_trials_data.correct_side = all_data.correct_side(ci);
completed_trials_data.response_side = all_data.response_side(ci);
completed_trials_data.completed_inds = ci;
completed_trials_data.traces_stim_aligned = traces_stim_aligned;
completed_trials_data.traces_resp_aligned = traces_resp_aligned;
frame_info = struct;
frame_info.stim_start_frame = all_data.frame_info.stim_start_frame(ci);
frame_info.start_poke_frame = all_data.frame_info.start_poke_frame(ci);
frame_info.stim_end_frame = all_data.frame_info.stim_end_frame(ci);
frame_info.response_frame = all_data.frame_info.response_frame(ci);
completed_trials_data.frame_info = frame_info;

data = struct;
data.all_data = all_data;
data.completed_trials_data = completed_trials_data;

save('all_data.mat','data')