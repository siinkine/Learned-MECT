function y = ForwardModel(object, W, M, D, noise, normalize)
%  y = ForwardModel(object, W, M, D, noisy, normalize)

% Compute forward projections:
projections = W * object;

% Compute photon counts as seen by the detector:
attenFactors = exp(- M * projections.');
counts = D * attenFactors; %Detector response model
countsNoAttn = D * ones(size(attenFactors));

if (noise) % Add poisson noise to the photon counts
	noisy_counts = poissrnd(counts);
else
	noisy_counts = counts;
end

if (normalize) %i.e Flat-field correction
	y = noisy_counts ./ countsNoAttn;
else
	y = noisy_counts;
end

y = y.';

end
