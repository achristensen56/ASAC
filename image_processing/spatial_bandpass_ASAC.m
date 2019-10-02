function data = spatial_bandpass_ASAC(data, radius, ...
    f_lower_scale, f_upper_scale, use_gpu)
% Butterworth bandpass filter for multi-dimensional images.

if ~exist('use_gpu', 'var')
    use_gpu = isa(data, 'gpuArray');
end
% If data is already on GPU, we deal differently
is_input_gpuArray = isa(data, 'gpuArray');

is_2d = ismatrix(data);

% Use same frequency scale in x,y
if is_2d
    [h, w] = size(data);
else
    [h, w, t] = size(data);
end
%nf = 2^nextpow2(max(h, w));
 nf = max(h, w);
% degree of the Butterworth polynomial
n = 4;

% Cutoff frequencies 
f_corr = nf / pi / radius;
f_lower = f_corr / f_lower_scale;
f_upper = f_corr * f_upper_scale;

% Create butterworth band-pass filter
[cx, cy] = meshgrid(1:nf, 1:nf);
dist_matrix = sqrt((cy - nf / 2).^2 + (cx - nf / 2).^2);
% Zero distance might cause a NaN in hpf, threshold it above zero
dist_matrix = max(dist_matrix, 1e-6);
lpf = 1 ./ (1 + (dist_matrix / f_upper).^(2 * n));
hpf = 1 - 1 ./ (1 + (dist_matrix / f_lower).^(2 * n));
bpf = single(lpf .* hpf);
bpf = maybe_gpu(use_gpu, bpf);

% If using GPU, chunk data in time so that we don't run out of memory
if use_gpu && ~is_2d
    slack_factor = 10;
    d = gpuDevice();
    available_space = d.AvailableMemory / 4 / slack_factor;
    n_chunks = ceil(t * nf^2 / available_space);
    chunk_size = ceil(t / n_chunks);
else
    %n_chunks = 10;
    chunk_size = 100;
    if is_2d
        chunk_size = 1;
    else
        %chunk_size = t/n_chunks;
        n_chunks = ceil(t/chunk_size);
    end
end

for i = 1:n_chunks
    idx_begin = (i - 1) * chunk_size + 1;
    idx_end = min(t, i * chunk_size);
    data_small = data(:, :, idx_begin:idx_end);
    % Send to GPU if use_gpu=true and data not already on GPU
    data_small = maybe_gpu(use_gpu & ~is_input_gpuArray, data_small);
    
    % Pad array up to nf
    data_small = padarray(data_small, [nf-h,nf-w],...
        0, 'post');

    % fft
    data_small = fft2(data_small);
    data_small = fftshift(data_small);

    % Filter and convert back to space
    data_small = bsxfun(@times, data_small, bpf);
    data_small = ifftshift(data_small);
    data_small = ifft2(data_small);
    if is_2d
        data_small = real(data_small(...
            1:end-(nf-h), 1:end-(nf-w)));
    else
        data_small = real(data_small(...
            1:end-(nf-h), 1:end-(nf-w), :));
    end

    % Update current chunk
    if use_gpu && ~is_input_gpuArray
        data_small = gather(data_small);
    end
    data(:, :, idx_begin:idx_end) = data_small;
end

%     if exist('div_norm', 'var') && div_norm
%         data(:, :, idx_begin:idx_end) = data(:, :, idx_begin:idx_end) ...
%             ./ data_small;
%     else
%         data(:, :, idx_begin:idx_end) = data_small;
%     end