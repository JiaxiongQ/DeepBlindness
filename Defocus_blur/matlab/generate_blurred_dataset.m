
function [] = blur_dataset()
    %  BLUR_DATASET generates synthetically blurred images when given pairs of RGB 
    %   and corresponding depth maps. This code is an adaptation of the layered
    %   approach proposed in "A layer-based restoration framework for variable 
    %   aperture photography", Hasinoff, S.W., Kutulakos, K.N., IEEE 11th 
    %   International Conference on Computer Vision, to create a realistic defocus
    %   blur:

    % Authors: Pauline Trouvé-Peloux and Marcela Carvalho.
    % Year: 2017

    % load parameters
    % Uncomment the next line to reproduce experiments from section 3 of the 
    % paper. Change focus on the file to make different tests.
    parameters_blurred_NYUv2;
    
    path = ['/share_data/public/yuxinyuan/kitti/defocus_R/outputRD.txt'];
    fpn = fopen(path,'rt');
    %path_depth = ['/share_data/public/yuxinyuan/kitti/dense_depth/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png'];
    %path_rgb = ['/share_data/public/yuxinyuan/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png'];

    %dest_path_rgb = ['/home/qiujiaxiong/d3net_depth_estimation/matlab/result.png'];

    %for i=1:(length(contents_rgb)-2)
    inx = 0;
     while feof(fpn) ~= 1
        inx = inx + 1
        if inx < 839
            continue;
        end
        line = fgetl(fpn);
        lines = strsplit(line,'&');
        path_rgb = char(lines{1});
        path_depth = char(lines{2});
        dest_path_rgb = strcat('/share_data/public/yuxinyuan/kitti/defocus_R/images/defocus_rgb_',num2str(inx),'.png');
        dest_path_gt = strcat('/share_data/public/yuxinyuan/kitti/defocus_R/gts/defocus_gt_',num2str(inx),'.png'); 
        im = double(imread(path_rgb));
        depth = imread(path_depth);
        
        if length(size(depth)) > 1
            depth = depth(:,:,1);
        end
        depth=double(depth)*10.0/255;
        %depth = medfilt2(depth,[5,5]);
        h=ones(5,5)/32; 
        h(3,3)=0;
        depth=imfilter(depth,h);
        size(depth)
        disp('begin......')
        r=8.*rand([1,1]);
        focusr = r
        [im_refoc, ~, ~, D,gt]=refoc_image(im,depth,step_depth,focusr,f,N,px,dmode);

        imwrite(uint8(im_refoc), dest_path_rgb)
        imwrite(uint8(gt), dest_path_gt)

end

%function [] = create_dir(dir_path)
%    if(exist(dir_path)~=7)
%        mkdir(dir_path)
%    else
%        display([dir_path  'already exists'])
%    end
%end
