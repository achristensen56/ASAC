v = VideoWriter('myFile2','Uncompressed AVI');
open(v);
M = M-min(min(min(M)));
M = M./(max(max(max(M))));
for i = 1:32876
    A = M(:,:,i);
    writeVideo(v,A);
end
close(v);



a = dir('./leah_20200212');
file_list = [];
for i = 3:length(a)
    file_list = [file_list str2num(a(i).name(7:end-4))];
end

[~,inds] = sort(file_list);
inds = [0 0 inds];
inds = inds+2;
v = VideoWriter('myFile3','Uncompressed AVI');
open(v);
a = dir('./leah_20200212');

for i = 3:length(a)
    if mod(i,100)==0
        i
    end
    A = imread(['.\leah_20200212\' a(inds(i)).name]);
    writeVideo(v,A);
    
end
close(v);


