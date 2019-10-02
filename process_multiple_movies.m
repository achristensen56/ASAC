files = dir('*201907*');

for i = 3:length(files)
    
    try
        pp_movie(files(i).name);
    catch
        'abort'
        i
        cd ..
    end
end