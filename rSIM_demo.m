% SIM = Tiff('NEIL_1_100N36_F_9C_interleaved_SIM_Reconstruction_noMatch_1C.tif','r');
% pWF = Tiff('NEIL_1_100N36_F_9C_interleaved_pWF_Reconstruction_noMatch_1C.tif','r');
% SIMData = read(SIM);
% pWFData = read(pWF);
% 
% 
% subplot(2,4,1)
% imshow(SIMData);
% title('bSIM')
% 
% subplot(2,4,5)
% imshow(pWFData);
% title('pWF')
% 
% 
% %% Compute 2D Fourier Transform
% 
% subplot(2,4,2)
% H_SIM = fft2(SIMData); 
% H_SIM = fftshift(H_SIM);
% f_SIM = log(H_SIM+1);
% imshow(f_SIM, [])
% title('FFT2(bSIM)')
% 
% subplot(2,4,6)
% H_pWF = fft2(pWFData); 
% H_pWF = fftshift(H_pWF);
% f_pWF = log(H_pWF+1);
% imshow(f_pWF, [])
% title('FFT2(pWF)')
% 
% %% Some FFT Maths
% 
% period_p = 50; %pixels (arbitrary for now) -> 1/kp
% kp = 1/period_p;
% crossover_scale = 0.15; %(will be used for crossover) 
% crossover_Freq = crossover_scale*kp;
% 
% % Initialize filters
% filter = zeros(584,579);
% imageSizeX = 579;
% imageSizeY = 584;
% [columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% 
% % Create the LPF 
% centerX = imageSizeX/2;
% centerY = imageSizeY/2;
% 
% radius = period_p*crossover_scale; % USing crossover and period
% 
% LPF = any((rowsInImage(:) - centerY).^2 + (columnsInImage(:) - centerX).^2 <= radius.^2, 2);
% LPF = reshape(LPF, imageSizeY, imageSizeX);
% subplot(2,4,3);
% imshow(LPF)
% title('bSIM - LPF')
% 
% 
% % HPF = 1-LPF
% HPF = abs(1-LPF);
% subplot(2,4,7)
% imshow(HPF);
% title('pWF - HPF')
% 
% % Apply LPF and plot IFFT
% LP_SIM = H_SIM.*LPF;
% LP_SIM_ifft = real(ifft2(ifftshift(LP_SIM)));
% subplot(2,4,4)
% imshow(LP_SIM_ifft, []); 
% title('LPF(bSIM)')
% 
% % Apply HPF and plot IFFT
% HP_pWF = H_pWF.*HPF;
% HP_pWF_ifft = real(ifft2(ifftshift(HP_pWF)));
% subplot(2,4,8)
% imshow(HP_pWF_ifft, []); 
% title('HPF(pWF)')
% 
% % Reconstruct in F, and apply IFFT 
% alpha = 1.0;
% rSIM_F = LP_SIM + alpha*HP_pWF;
% rSIM_IFFT = real(ifft2(ifftshift(rSIM_F)));
% 
% % Save rSIM IFFT
% imwrite(uint16(rSIM_IFFT),'rSIM_Reconstruction.tif')
% 
% 


%% Now for stacks 
fname_pWF = '090823\\100N36_F\\NEIL_1_100N36_FC_interleaved_pWF_Reconstruction_noMatch.tif';
fname_SIM = '090823\\100N36_F\\NEIL_1_100N36_FC_interleaved_SIM_Reconstruction_noMatch.tif';

[pWFData, yPix, xPix, tPoints] = read_file(fname_pWF);
[SIMData, ~, ~, ~] = read_file(fname_SIM);

rSIM_stack = zeros(size(pWFData));
for i=1:tPoints
    rSIM_stack(:,:,i) = rSIM(pWFData(:,:,i), SIMData(:,:,i), 50, 0.15, 1.0, xPix, yPix);
end

bfsave(rSIM_stack, 'rSIM_Test.tiff')



function [imgstack, xPixels, yPixels, tPoints] = read_file(fileName)
    Data = bfopen(fileName);
    val = Data{1,1};
    val2 = val(:,1);
    [xPixels,yPixels] = size(val2{1,1});
    tPoints = length(val2);
    imgstack = zeros(xPixels, yPixels, tPoints, 'uint16');
    for i=1:tPoints
        imgstack(:,:,i)=cell2mat(val2(i));
    end
    
end


function r = rSIM(pWFData,SIMData, period_p, crossover_scale, alpha,xPix,yPix)
    
    % take FFT's
    H_SIM = fft2(SIMData); 
    H_SIM = fftshift(H_SIM);
    H_pWF = fft2(pWFData); 
    H_pWF = fftshift(H_pWF);

    % Initialize filters
    imageSizeX = xPix;
    imageSizeY = yPix;
    [columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);

    % Create the Filters
    centerX = imageSizeX/2;
    centerY = imageSizeY/2;
    radius = period_p*crossover_scale; % USing crossover and period
    LPF = any((rowsInImage(:) - centerY).^2 + (columnsInImage(:) - centerX).^2 <= radius.^2, 2);
    LPF = reshape(LPF, imageSizeY, imageSizeX);
    HPF = abs(1-LPF);

    % Apply Filters
    LP_SIM = H_SIM.*LPF;
    HP_pWF = H_pWF.*HPF;

    % Reconstruct in F and then IFFT
    rSIM_F = LP_SIM + alpha*HP_pWF;
    r = real(ifft2(ifftshift(rSIM_F)));

end

























