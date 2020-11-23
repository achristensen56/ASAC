ds1 = DaySummary('', ''); % adam0_20190624 ; adam0_20190731 % adam0_20190228 % anton120190302 % terry20190216 %adam0_20190516 (task) % terry20190216 %adam0_2019303 (task) % adam0_20190812b
choices1 = load('choices.mat'); choices1 = choices1.choices;


for i = 1:length(ds1.cells)
    if choices1(i)==1
        ds1.cells(i).label = 'cell';
    else
        ds1.cells(i).label = 'not a cell';
    end
end

ds2 = DaySummary('', ''); % 0704 ; adam0_20190729 % adam0_20190301 % anton1_20190228          % terry20190214 %adam0_20190302 (tune) % terry20190214 %adam0_20190302 (tune) % adam0_20190731 (tune_
choices2 = load('choices.mat'); choices2 = choices2.choices;


for i = 1:length(ds2.cells)
    if choices2(i)==1
        ds2.cells(i).label = 'cell';
    else
        ds2.cells(i).label = 'not a cell';
    end
end

ds3 = DaySummary('',''); %07 29
ds4 = DaySummary('','');

[match_1to2, match_2to1] = run_alignment(ds1, ds2);
[match_2to3, match_3to2] = run_alignment(ds2, ds3);
%[match_3to4, match_4to3] = run_alignment(ds3, ds4);

ds_list = {1, ds1; 2, ds2};
match_list = {1, 2, match_1to2, match_2to1};
md = MultiDay(ds_list, match_list);


ds_list = {1, ds1; 2, ds2; 3, ds3};
match_list = {1, 2, match_1to2, match_2to1; 2, 3, match_2to3, match_3to2};
md = MultiDay(ds_list, match_list);

browse_multiday(md)

combined_traces = get_multiday_traces(md);

total_frames = [ds1.full_num_frames ds2.full_num_frames ds3.full_num_frames];

save('combined_traces.mat','combined_traces');
save('total_frames.mat','total_frames');
save('multiday_variables.mat');


ds_list = {1, ds1; 2, ds2};
match_list = {1, 2, match_1to2, match_2to1};
md = MultiDay(ds_list, match_list);

browse_multiday(md)

combined_traces = get_multiday_traces(md);

total_frames = [ds1.full_num_frames ds2.full_num_frames ds3.full_num_frames];

save('combined_traces.mat','combined_traces');
save('total_frames.mat','total_frames');
save('multiday_variables.mat');

k = 0;
for i = 1:length(choices1)
    if choices1(i)
        choices1(i,2) = k;
        k = k+1;
    end
end

k = 0;
for i = 1:length(choices2)
    if choices2(i)
        choices2(i,2) = k;
        k = k+1;
    end
end

match_inds = [];
k=1;
for i = 1:length(md.matched_indices)
    ind1 = md.matched_indices(i,1);
    ind2 =  md.matched_indices(i,2);
    if choices1(ind1,1) && choices2(ind2,1)
        match_inds(k,1) = choices1(ind1,2);
        match_inds(k,2) = choices2(ind2,2);
        k = k+1;
    end
end

task_inds = match_inds(:,1);
tune_inds = match_inds(:,2);

save('task_inds.mat','task_inds');
save('tune_inds.mat','tune_inds');
save('multiday_variables.mat');


MI = md.matched_indices;

for i = 1:length
    
end