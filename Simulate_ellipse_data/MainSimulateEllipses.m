function MainSimulateEllipses()
% MainSimulateEllipses()
%
% This is the main function which simulates the PCD projection data for
% random ellipses phatoms.
%
%
% 27.3.2020 Satu Inkinen
% 2.8.2020 SI: code cleaned
%--------------------------------------------------------------------------

clear, clc
s = rng(1561354);
addpath('functions');

%% Parameters for ellipse phantom:
param.n = 1000; %number of samples
param.imSize = 256; %image size
param.maxNumMat = 10; %maximum number of materials in one slice

%% Generate random phantom geometries: 
%Load attenuation table which contains the linear attenuation coefficents
%for various soft tissue and contrast materials:
load('simulator_data/attn_table_corrected.mat');

% load material densities:
load(fullfile('simulator_data','DENSITIES.MAT'), 'densities'); 

%Simulated image pixels contains the index of the attenuation table material
indexVals = [2:36];

% Exclude heavy elements:
exludeList = [20, 22, 23, 25, 26, 29];
for kk = 1:length(exludeList)
    idx = find(indexVals == exludeList(kk));
    indexVals(idx) = [];
end

nEllipses = randi(param.maxNumMat,param.n,1); %Randomly sample for each slice the number of ellipses
phantomSlices = zeros(param.imSize, param.imSize, param.n); %the generated slices are added to 3rd dimension

for ii = 1:param.n %loop through each slice
    
    materials = randi(length(indexVals), nEllipses(ii),1 ); %Randomly sample material for each ellipse

    %Ellipse parameters: 
    scaleEllipseSize = 10;
    radiusY = round(param.imSize/scaleEllipseSize +  scaleEllipseSize*randn(nEllipses(ii), 1)); %randi(256/2,nEllipses(ii),1);
    radiusX = round(param.imSize/scaleEllipseSize +  scaleEllipseSize*randn(nEllipses(ii), 1));%randi(256/2,nEllipses(ii),1);
    centerPoints = randi(param.imSize,nEllipses(ii),2);
    
    im = zeros(param.imSize,param.imSize); %Initialize image
    for jj = 1:nEllipses(ii) %Loop though ellipses
        
        ellipseMask = makeEllipseMask(im, centerPoints(jj, :),radiusX(jj), radiusY(jj));
        idx = find(ellipseMask == 1);
        %Assign ellipse to image with material value:
        im(idx) =  indexVals(materials(jj));
        
    end
    
    ellipseMaskOuter = makeEllipseMask(im, [param.imSize/2, param.imSize/2],param.imSize/2, param.imSize/2);
    
    im = im.*ellipseMaskOuter; %circular mask to avoid interior tomography
    
    im(im == 0) = 24; %index for air

    %figure, imshow(im,[])
    
    
    %Assign data:
    phantomSlices(:,:, ii) = im;
    
    if mod(ii,100)==1
        disp([num2str(ii), ' / ', num2str(param.n)]);
    end
end

disp('Geometries generated!');


%% Load simulated response data for PCD:
load(fullfile('simulator_data', 'ME3_response_20_150keV.mat'));
%Re-bin response data to 1 keV increments:
Ebin = 1; %keV increments

responses = [];
figure, hold on
for ii = 1:length(dataStruct)
    EVALS = dataStruct(ii).EVALS;
    E = min(EVALS(:)):Ebin:max(EVALS);
    N = dataStruct(ii).numPhotons;
    N = interp1(EVALS,N,E,'linear');
    dataStruct(ii).EVALS = E;
    dataStruct(ii).numPhotons = N*0.5; %*1e2;    
    
    responses(ii,:)  = dataStruct(ii).numPhotons;

    plot(dataStruct(ii).EVALS, dataStruct(ii).numPhotons, 'LineWidth', 1.5)
    xlabel('Energy (keV)')
    ylabel('Intensity (Arb. unit.)')
    title('Responses 119 spectral channels')
end


%%  Simulation geometry using ASTRA in 2D:

%Geometry params:
projParam.NbProj = 37; %Number of projections
projParam.angles = linspace2(0,2*pi,projParam.NbProj);

projParam.cols = param.imSize;
projParam.numPixels = param.imSize; % Plane detector array of 2 ME100 modules of 128 pixels
projParam.pixelSize = 0.077;     % Pixel size [cm]
projParam.gapsize = 0.153;       % Gap size between the 2 modules
projParam.SDD = 115.55;          % Distance source to detector
projParam.SAD = 57.50;           % Distance between source and rotation axis

projParam.det_count = projParam.numPixels;
projParam.source_origin = projParam.SAD/projParam.pixelSize;  %distance between the source and the center of rotation
projParam.origin_det =   (projParam.SDD-projParam.SAD)/projParam.pixelSize; %distance between the center of rotation and the detector array
projParam.det_width = 1;

%Parallel beam geometry:
proj_geom = astra_create_proj_geom('parallel', projParam.det_width, projParam.det_count, projParam.angles);
vol_geom  = astra_create_vol_geom(projParam.numPixels,projParam.numPixels);

% Generate projection data:
W = opTomo('cuda', proj_geom, vol_geom); % Create the Spot operator for ASTRA using the GPU.

%%  Simulate projection data:
noise = true; % Simulate noisy projections

h = waitbar(0,'Generate projections');
for ii = 1:param.n
    
    x =  phantomSlices(:,:, ii);
    
    materials = unique(x(:)); %list of materials
    [materials I] = sort(materials, 'ascend');
    
    M = table2array(attn_table_corr(:,materials));
    density = densities(materials);
    
    %Bin material attenuations to same as the detector model:
    M = M(40:2:2*150,:)./repmat(density, [size(responses,2), 1]); 
    E  = table2array(attn_table_corr(40:2:2*150,1));
    %Vectorize phantom:
    xVec = zeros([size(x, 1)*size(x, 2), length(materials)]); %initialize
    for jj = 1:length(materials) %loop trough materials
        
        temp_mask = x == materials(jj);
        xVec(:, jj)  = temp_mask(:).*densities(materials(jj));   %density g/cm^3
        
    end
    
    xVec = projParam.pixelSize*xVec; %g/cm^3 * cm = g/cm^2
    
    % Apply forward model:
    y = ForwardModel(xVec, W, M, responses, noise, 1);
    y = -log(y);

    %Ground truth data:
    X = optimalPhantom(xVec, M, responses, E);
    X = reshape(X, [param.imSize, param.imSize, size(responses, 1)]);

    % Store results:
    savePath = '2D_data_final';
    saveName = ['data', num2str(ii), '.mat'];
    size_info = [projParam.pixelSize, projParam.NbProj, vol_geom.GridColCount, size(y, 2)];
    save(fullfile(savePath , saveName), 'X', 'y', 'size_info');
    waitbar(ii/param.n,h)
    
end
close(h);


disp('Done!');

end