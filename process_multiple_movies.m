files = dir('*_2019*');

for i = 3:length(files)
    ff = ['./' files(i).name '/*dff*'];
    a = dir(ff);
    
    if length(a) == 0
        try
            files(i).name
            pp_movie(files(i).name);
        catch
            'abort'
            i
            cd ..
        end
    end
end