fid = fopen( 'trialdata.txt', 'wt' );

for i = find(temp_obj.trial_completed == 1)
    start_arm = 'north'; goal_arm = 'west'; end_arm = 'west';
    if temp_obj.stim_right(i)==1
        goal_arm = 'east';
    end
    if temp_obj.response_right(i)==1
        end_arm = 'east';
    end
    
    start_frame = temp_obj.start_poke_frame(i)-5;
    open_gate_frame = temp_obj.stim_start_frame(i);
    close_gate_frame = temp_obj.end_poke_frame(i);
    end_frame = temp_obj.reward_frame(i)+5;
    
    duration_of_trial = (end_frame - start_frame)/10;
    duration_of_trial =10;
    fprintf( fid, '%s %s %s %f %d %d %d %d\n', start_arm, goal_arm, end_arm, duration_of_trial,start_frame,open_gate_frame,close_gate_frame,end_frame);
end


fclose(fid);
