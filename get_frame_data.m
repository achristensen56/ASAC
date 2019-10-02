function [start_poke_frame, stim_start_frame, end_poke_frame, reward_frame, stim_end_frame,trial_completed] = get_frame_data(obj,fps)
    % pull out data for synchronizing to imaging
    
    obj = obj.obj;
    %% get last full nosepoke (no timeout)
    start_poke_frame = [];
    end_poke_frame = [];
    for i = 1:length(obj.response.trial_initiation.start_poke_frame)
        start_poke_frame = [start_poke_frame obj.response.trial_initiation.start_poke_frame{i}(end)];
        end_poke_frame = [end_poke_frame obj.response.trial_initiation.end_poke_frame{i}(end)];
    end

    %% get reward time
    reward_frame = obj.response.stim_response.response_frame;
    stim_end_frame =  obj.response.stim_response.stim_end_frame;
    trial_completed = -1*((obj.response.stim_response.response_side==0)-1);



    %% fix roll around
    % ROLL AROUND WORKS BY GOING FROM 32767 TO -32768
    for i = find(diff(start_poke_frame)<0)
        start_poke_frame(i+1:end) =  start_poke_frame(i+1:end) + 32768 + 32767 + 1;
        end_poke_frame(i+1:end) = end_poke_frame(i+1:end) + 32768 + 32767 + 1;
        reward_frame(i+1:end) = reward_frame(i+1:end) + 32768 + 32767 + 1;
        stim_end_frame(i+1:end) = stim_end_frame(i+1:end) + 32768 + 32767 + 1;

    end

    %% compute stim start frame
    stim_start_frame = start_poke_frame + round(obj.behavior_params.minCenterTime*fps);

   end

% obj.response.stim_response.response_frame
% obj.response.stim_response.start_frame
% plot(obj.response.trial_initiation.start_frame)
% hold on
% plot(obj.response.stim_response.start_frame)
% plot(obj.response.trial_start_frame)
% plot(obj.response.trial_initiation.start_frame)
% plot(obj.response.trial_initiation.end_frame)
% plot(start_poke_frame)
% hold on
% plot(end_poke_frame)
%
% figure
% plot(end_poke_frame - start_poke_frame)