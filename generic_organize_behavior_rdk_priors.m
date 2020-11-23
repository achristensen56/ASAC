clear all
load('terry_session1.mat');

all_data = struct;

n_trials = 1:(find(obj.is_trial_completed,1,'last')-1);
all_data.was_completed = obj.is_trial_completed(n_trials);
all_data.was_correct = obj.response.stim_response.response_correct(n_trials);
all_data.stim_dir = obj.dots.dirs(n_trials);
all_data.correct_side = obj.behavior_params.correct_side(n_trials);
to_mult = all_data.correct_side;to_mult(to_mult==3) = -1;
all_data.noise = obj.prob_params.coherence(n_trials).*to_mult;
all_data.response_side = obj.response.stim_response.response_side(n_trials);
all_data.traces = load('traces.mat'); all_data.traces = all_data.traces.traces;
all_data.task_info = obj.dots;
all_data.prior = obj.prob_params.close_priors(n_trials);
frame_info = struct;


frame_info.stim_start_frame = obj.response.stim_response.start_frame(n_trials);
frame_info.stim_start_frame(frame_info.stim_start_frame<0) = frame_info.stim_start_frame(frame_info.stim_start_frame<0) + 65536;

frame_info.start_poke_frame = cellfun(@(x) x, obj.response.trial_initiation.start_poke_frame);
frame_info.start_poke_frame(frame_info.start_poke_frame<0) = frame_info.start_poke_frame(frame_info.start_poke_frame<0) + 65536;

frame_info.stim_end_frame = obj.response.stim_response.stim_end_frame(n_trials);
frame_info.stim_end_frame(frame_info.stim_end_frame<0) = frame_info.stim_end_frame(frame_info.stim_end_frame<0) + 65536;

frame_info.response_frame = obj.response.stim_response.response_frame(n_trials);
frame_info.response_frame(frame_info.response_frame<0) = frame_info.response_frame(frame_info.response_frame<0) + 65536;

all_data.frame_info = frame_info;


completed_trials_data = struct;



% MAKE STIM AND RESPONSE ALIGNED NEURAL DATA
traces = all_data.traces';
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
completed_trials_data.noise = all_data.noise(ci);
completed_trials_data.prior = all_data.prior(ci);
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

