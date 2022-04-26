clc
close all
clear all
model_mode = 'post';    %选择SR模型框架，'pre'为预上采样SR框架，'post'为后上采样SR框架
downsample_mode = 'BI'; %选择下采样方法，'BI'为仅采用双三次插值法，'BD'为高斯模糊下采样
del_LR = true;          %在LR图像已经存在时选择是否删除LR图像，为true则删除LR图像并重新生成，为false则直接跳过相关文件夹的LR图像制作
scale_list = [2, 3, 4];
folder_list = dir('./data/');
check(model_mode, downsample_mode);
for i = 3 : length(folder_list)
    data_folder = ['./data/', folder_list(i).name, '/'];
    dataset_list = dir(data_folder);
    dataset_list = delete_LR_folder(data_folder, dataset_list, del_LR);
    for j = 3 : length(dataset_list)
        dataset = [data_folder, dataset_list(j).name];
        LR_folder = [dataset, '_LR/'];
        mkdir(LR_folder);
        dataset = [dataset, '/'];
        disp(dataset_list(j).name)
        for s = 1 : length(scale_list)
            scale = scale_list(s);
            LR_scale_folder = [LR_folder, num2str(scale), '/'];
            mkdir(LR_scale_folder);
            img_path = dir(fullfile(dataset,'*.*'));
            parfor num = 3 : length(img_path)
                [add, imname, type] = fileparts(img_path(num).name);
                hr = imread([dataset imname type]);
                lr = downsample(hr, model_mode, downsample_mode, scale);
                LR_path = [LR_scale_folder, imname , type];
                disp(LR_path)
                imwrite(lr, LR_path);
            end
        end
    end
end

function check(model_mode, downsample_mode)
    if ~(strcmp(downsample_mode, 'BI') || strcmp(downsample_mode, 'BD'))
        error('WRONG downsample_mode! It can only be set to BI or BD!')
    end
    if ~(strcmp(model_mode, 'pre') || strcmp(model_mode, 'post'))
        error('WRONG model_mode! It can only be set to pre or post!')
    end
end

function dataset_list = delete_LR_folder(data_folder, dataset_list, del_LR)
    del_list = [];
    del_HR_list = [];
    for j = 3 : length(dataset_list)
        LR_find = strfind(dataset_list(j).name, '_LR'); %查找是否存在LR图像
        Y_find = strfind(dataset_list(j).name, '_Y');   %查找是否存在Y通道图像
        LR_find = [LR_find, Y_find];
        LR_find = sort(LR_find);
        if ~isempty(LR_find)
            del_list(end + 1) = j;
            del_dataset = strrep(dataset_list(j).name, '_LR', '');
            while strfind(dataset_list(j).name, del_dataset)
                j = j - 1;
            end
            del_HR_list(end + 1) = j + 1;
        end
    end
    if ~del_LR
        del_list = unique(sort([del_list, del_HR_list]));
    end
    for d = length(del_list) : -1 : 1
        if del_LR
            rmdir([data_folder, dataset_list(del_list(d)).name, '/'],'s');
        end
        dataset_list(del_list(d)) = [];
    end
end

function lr = shave(lr, hr)
    [h_h, w_h, ~] = size(hr);  %裁剪或扩充LR使得LR和HR尺寸一致
    [h_l, w_l, ~] = size(lr);
    if h_h < h_l
        lr = lr(1:h_h, :, :);
    elseif h_h == h_l
    else
        h_end = h_h - h_l;
        lr = lr(end + h_end, :, :);
    end
    if w_h < w_l
        lr = lr(:, 1:w_h, :);
    elseif w_h == w_l
    else
        w_end = w_h - w_l;
        lr = lr(:, end + w_end, :);
    end
end

function lr = downsample(hr, model_mode, downsample_mode, scale)
    if strcmp(downsample_mode, 'BD')
        G = fspecial('gaussian', [5 5], 2);
        hr = imfilter(hr, G, 'same');
    end
    lr = imresize(hr, 1/scale, 'bicubic');
    if strcmp(model_mode, 'pre')
        lr = imresize(lr, scale, 'bicubic');
        lr = shave(lr, hr);
    end
end