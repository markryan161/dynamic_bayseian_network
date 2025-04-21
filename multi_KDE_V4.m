function [pdf_grid, g1] = multi_KDE_V4(samples)
%% MULTIVARIATE KDE V4
% CROSS VALIDATION V4 Model
% K-Means model with local evaluation points
% Tested with 3^d folds
% Only worth using on 3d+


D = size(samples,2);
N = length(samples);
grid_res = ones(1,D)*N;

eval_points = adaptive_grid_V2(samples,20,50);
SIGMA = sample_covariance(samples);

% Bandwidth Calculation
epsilon = 1e-6;
H = SILVERMAN(samples,SIGMA);
H = (H + epsilon * eye(size(H)));

pdf = zeros(length(eval_points),1);
normal_factor = 1 / ((2 * pi)^(D / 2) * sqrt(det(H)));

for i = 1:length(eval_points)

    diff = samples - eval_points(i,:);
    pdf(i) = sum(exp(-0.5 * sum((diff / H) .* diff, 2)));

end

pdf = normal_factor*pdf/N;
k = 4;
mins = min(eval_points, [], 1);
maxs = max(eval_points, [], 1);

g1 = linspace(mins(1), maxs(1), N);
g2 = linspace(mins(2), maxs(2), N);
g3 = linspace(mins(3), maxs(3), N);
g4 = linspace(mins(4), maxs(4), N);

[G1, G2, G3, G4] = ndgrid(g1, g2, g3, g4);
grid_points = [G1(:), G2(:), G3(:), G4(:)];

% Create KD-tree and search for k-nearest neighbors
Mdl = KDTreeSearcher(eval_points);
[idxs, dists] = knnsearch(Mdl, grid_points, 'K', k);

% Inverse-distance weighting
weights = 1 ./ (dists.^2 + eps);
weights = weights ./ sum(weights, 2);

% Interpolate
pdf_interp = zeros(size(grid_points, 1), 1);
for i = 1:k
    pdf_interp = pdf_interp + pdf(idxs(:,i)) .* weights(:,i);
end
pdf_interp = pdf_interp/sum(pdf_interp(:));
% Reshape to 4D grid
pdf_grid = reshape(pdf_interp, grid_res);



end


%% SILVERMAN

function H = SILVERMAN(samples,covariance)
d = width(samples);
N = length(samples);
sigma = covariance;

H = ((4/(d+2))^(1/(d+4))*N^(-1/(d+4)))*sigma;
end
