function full_movie = load_tif_from_directory(file_prepends, varargin)
% Usage: M = load_movie_from_tif('mouse1_day4_sp2_mc_cr.tif');
%
% Optional keyword 'usexml' will open the corresponding XML file (i.e. for
% Miniscope recordings) and fill in dropped frames.
%
% 2016.09.14 - Adam Shai. This is a modification of the loading file for
% our purposes. It reads through the xml files to get the number of frames
% per trial, initializes a matrix to load the files into one by one.
% Example:
% file_prepend = './recording_20160906*', in order to get all
% recordings taken on that date.

use_xml = 0;

%the default downsampling in space is 1/2,  I need  to learn to use
%varargin to make that not the case. -AmyJC
%If downsampling, we might want to preprocess while loading the movie. 
%currently preprocess will apply a 3x3 medfilt to each frame as it's
%loaded. 

downsample = 0;
preprocess = 0;
for k = 1:length(varargin)
    if ischar(varargin{k})
        vararg = lower(varargin{k});
        switch vararg
            case {'xml', 'usexml'}
                use_xml = 1;
            case {'downsample'}
                downsample = 1;
            case {'preprocess'}
                preprocess = 1;
        end
    end
end

% get names of all tif files
%tif_files = dir([directory '/recording_20160906*.tif']); % filenames are tif_file(i).name

xml_files = dir([file_prepends '.xml']); % filenames are tif_file(i).name

trial_frames = [0];
tif_files = [];
for i = 1:length(xml_files)
    dum = parse_miniscope_xml(xml_files(i).name);
    tif_files = [tif_files;dum.files];
    
    for j = 1:length(dum.files)
       dum2 = xml2struct(xml_files(i).name);
       if length(dum.files)>1
            trial_frames = [trial_frames str2num(dum2.recording.decompressed.file{j}.Attributes.frames)];
       else
           trial_frames = [trial_frames str2num(dum2.recording.decompressed.file.Attributes.frames)];
       end
    end
end

info = imfinfo(tif_files{1});
tif_type = info(1).SampleFormat;

switch tif_type
    case 'Unsigned integer'
        type = 'uint16';
    case 'IEEE floating point'
        type = 'single';
    otherwise
        error('load_movie_from_tif: Unrecognized type "%s"\n', tif_type);
end


width  = info(1).Width;
height = info(1).Height;

if downsample
    width = floor(width / 2);
    height = floor(height / 2);
end

full_movie = zeros(height,width,sum(trial_frames),type);

total_frame = 0;    
for i = 1:length(tif_files)
    tic;
    fprintf('\n\n\n File %d of %d.',i,length(tif_files));
    source = tif_files{i};
    info = imfinfo(source);
    num_tif_frames = length(info);
    
    
    fprintf('\n Loading movie %d (%d frames) into memory...',i,num_tif_frames);
    % Load movie into memory
    %movie = zeros(height, width, num_tif_frames, type);
    t = Tiff(source, 'r');
    for k = 1:num_tif_frames
        if (mod(k,1000)==0)
            fprintf('  Frames %d / %d loaded\n', k, num_tif_frames);
        end
        t.setDirectory(k);
        temp = t.read();
        
        if downsample
            temp = temp(1:2:end, 1:2:end, :) + temp(2:2:end, 2:2:end, :);
        end
        
        if preprocess
            temp = medfilt2(temp);
        end
        
        full_movie(:,:,k + total_frame) = temp;
    end
    t.close();
    
    total_frame = total_frame + num_tif_frames;
    fprintf('\n Done loading file %d.',i);
    
    % Optionally, check XML for dropped frame correction.
%     % This is currently broken.
%     if use_xml
%         fprintf('\n Correcting for dropped frames');
%         xml_filename = convert_extension(source, 'xml');
%         xml_struct = parse_miniscope_xml(xml_filename);
% 
%         % Miniscope XML file tallies recorded and dropped frames separately.
%         % Make sure that the recorded frames match what is in the TIF file.
%         num_recorded_frames = str2double(xml_struct.frames);
%         assert(num_recorded_frames == num_tif_frames,...
%             '  Unexpected number of frames in TIF file!');
% 
%         num_dropped_frames = str2double(xml_struct.dropped_count);
%         if (num_dropped_frames ~= 0)
%             dropped_frames = str2num(xml_struct.dropped); %#ok<ST2NM>
% 
%             num_total_frames = num_recorded_frames + num_dropped_frames;
%             movie_corr = zeros(height, width, num_total_frames, type);
% 
%             % Each missing frame slot will be replaced with the PREVIOUS
%             % recorded frame. Except when the first frames of a recording are
%             % dropped; in that case, we replace with the SUBSEQUENT recorded
%             % frame.
%             tif_idx = 0;
%             for k = 1:num_total_frames
%                 if ismember(k, dropped_frames) % Is a dropped frame
%                     sample_idx = max(1, tif_idx); % Special handling for first frame
%                     movie_corr(:,:,k) = movie(:,:,sample_idx);
%                 else
%                     tif_idx = tif_idx + 1;
%                     movie_corr(:,:,k) = movie(:,:,tif_idx);
%                 end
%             end
%             assert(tif_idx == num_recorded_frames,...
%                 '  Not all recorded frames have been transferred to dropped-frame corrected movie!');
% 
%             movie = movie_corr;
%         end
%     end
    
    
    
    %just load directly into the movie, to save memory space. We can add a
    %check later for the usexml case. 
    %fprintf('\n Concatenating movie');
    %full_movie(:,:, (c_trial_frames(i)+1):c_trial_frames(i+1)) = movie; 
    fprintf('\n Done with file %d in %f seconds.',i,toc);
end