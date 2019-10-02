

% 5 before nosepoke
% 5 after nosepoke
% 10 during stim
% 5 during response
% 10 during reward


[start_trial_frame,response_frame] = get_gpio_frame_data(ls('*.csv')); stf = start_trial_frame; rf = response_frame;
obj = load(ls('*session1.mat')); obj = obj.obj;
load traces.mat;

total_frames = 35;
mct = obj.behavior_params.minCenterTime*10;
mtv = obj.behavior_params.min_time_vis*10;

n_data = zeros(size(traces,1),total_frames,obj.completed_trials); % neural data, NEURONS x FRAMES x TRIALS
v_data = []; % relevant variable data
% 
% k = 1;
% for i = 1:obj.curr_trial-1
%     
%     if obj.is_trial_completed(i)
%         
%         n_data(:,:,k)=traces(:,(start_trial_frame(i)-frame_before_poke):(start_trial_frame(i)+frame_after_poke));
%         
%         
%         v_data = [v_data obj.dots.gratings.direction(i)];
%         k = k+1;
%     end
%     
%     
% end
% 
% n_data = zeros(size(traces,1),total_frames,obj.completed_trials); % neural data, NEURONS x FRAMES x TRIALS

k = 1;
for i = 1:obj.curr_trial-1
    
    if obj.is_trial_completed(i)
        
        % 5 before nosepoke = stf-5:stf-1 [5 frames]
        % 5 after nosepoke = stf:(stf+mct-1) [interpolate to 5 frames]
        % 10 during stim = stf+mct:stf+mct+mtv-1 [interpolate to 10 frames]
        % 5 during response stf+mct_+mtv:rf-1 [interpolate to 5 frames]
        % 10 during reward rf:rf+10 [interpolate to 10 frames]
        
        
        % BEFORE NOSEPOKE
         bn = traces(:,stf(i)-5:stf(i)-1)'; % this is time by neuron
        
        % PRE STIMULUS
         ps = traces(:,stf(i):(stf(i)+mct-1))';
         ps = interp1(ps,linspace(1,size(ps,1),5));
        
         % DURING STIM
         ds = traces(:,(stf(i)+mct):(stf(i)+mct+mtv-1))';
         ds = interp1(ds,linspace(1,size(ds,1),10));
         
         % DURING RESPONSE MOVING
         rm = traces(:,(stf(i)+mct+mtv):(rf(i)-1))';
         rm = interp1(rm,linspace(1,size(rm,1),5));
         if size(rm,2)<size(ds,2)
            'hi'
            rm = traces(:,(stf(i)+mct+mtv):(rf(i)-1))';
            rm = [rm;rm;rm;rm;rm];
         end
         
         % DURING REWARD
         rd = traces(:,(rf(i):(rf(i)+9)))';
         rd = interp1(rd,linspace(1,size(rd,1),10));    
         

        n_data(:,:,k)=[bn;ps;ds;rm;rd]';
        
        
        v_data = [v_data obj.dots.gratings.direction(i)];
        k = k+1;
    end
    
    
end



% take trial averages according to relevant variable
n_av = zeros(size(traces,1),total_frames,length(unique(v_data))); % trial averaged neural data
k = 1;
for i = unique(v_data)
    v_inds = find(v_data==i);
    n_av(:,:,k)=nanmean(n_data(:,:,v_inds),3);
    k=k+1;
end

v_num = length(unique(v_data)); % number of values for thing
% 
% % make roseplots
% frames_to_av = 20:25;
% theta = unique(v_data)*pi/180; theta(end+1) = theta(1);
% for n = 1:size(n_av,1) % for each cell
%    figure;
%    dum = mean(squeeze(n_av(n,frames_to_av,:)),1); dum(v_num+1) = dum(1);
%    polarplot(theta,dum);
% end

% 
% plot trial averages
for n = 1:size(n_av,1) % for each cell
   figure;
   plot(squeeze(n_av(n,:,:)));
end

% compute OSI and DSI for PRESTIM
frames_to_av = 5:9;
osi = []; dsi = [];
for n = 1:size(n_av,1) % for each cell
   dum = nanmean(squeeze(n_av(n,frames_to_av,:)),1); 
   pR_ind = find(dum==max(dum)); oR_ind = mod([pR_ind + 2,pR_ind+6],v_num); op_ind = mod(pR_ind+4,v_num);
   if sum(oR_ind==0)
       oR_ind(oR_ind == 0) = 8;
   end
   
  if op_ind==0
   op_ind = 8;
  end
      
  osi = [osi (dum(pR_ind) - mean(dum(oR_ind)) )/dum(pR_ind)]; dsi = [dsi (dum(pR_ind) - dum(op_ind))/dum(pR_ind)];
   
end

figure;
subplot(3,2,1)
hist(dsi,0:.1:1); title('DSI PRESTIM'); xlim([0 1])

subplot(3,2,2)
hist(osi,0:.1:1); title('OSI PRESTIM'); xlim([0 1])


% compute OSI and DSI for stim period
frames_to_av = 20:24;
osi = []; dsi = [];
for n = 1:size(n_av,1) % for each cell
   dum = nanmean(squeeze(n_av(n,frames_to_av,:)),1); 
   pR_ind = find(dum==max(dum)); oR_ind = mod([pR_ind + 2,pR_ind+6],v_num); op_ind = mod(pR_ind+4,v_num);
   if sum(oR_ind==0)
       oR_ind(oR_ind == 0) = 8;
   end
   
  if op_ind==0
   op_ind = 8;
  end
      
  osi = [osi (dum(pR_ind) - mean(dum(oR_ind)) )/dum(pR_ind)]; dsi = [dsi (dum(pR_ind) - dum(op_ind))/dum(pR_ind)];
   
end

subplot(3,2,3)
hist(dsi,0:.1:1); title('DSI STIM'); xlim([0 1])

subplot(3,2,4)
hist(osi,0:.1:1); title('OSI STIM'); xlim([0 1])



%%%%%%%%%%%%
% compute OSI and DSI for RESPONSE period
frames_to_av = 25:30;
osi = []; dsi = [];
for n = 1:size(n_av,1) % for each cell
   dum = nanmean(squeeze(n_av(n,frames_to_av,:)),1); 
   pR_ind = find(dum==max(dum)); oR_ind = mod([pR_ind + 2,pR_ind+6],v_num); op_ind = mod(pR_ind+4,v_num);
   if sum(oR_ind==0)
       oR_ind(oR_ind == 0) = 8;
   end
   
  if op_ind==0
   op_ind = 8;
  end
      
  osi = [osi (dum(pR_ind) - mean(dum(oR_ind)) )/dum(pR_ind)]; dsi = [dsi (dum(pR_ind) - dum(op_ind))/dum(pR_ind)];
   
end

subplot(3,2,5)
hist(dsi,0:.1:1); title('DSI RESP'); xlim([0 1])

subplot(3,2,6)
hist(osi,0:.1:1); title('OSI RESP'); xlim([0 1])







figure;
plot(squeeze(nanmean(squeeze(nanmean(n_data,3)),1)))