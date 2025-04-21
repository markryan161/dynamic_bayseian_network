function [pdf, evaluation_points, c] = multi_KDE_V3(samples)
%% MULTIVARIATE KDE
% CROSS VALIDATION V1 Model


D = size(samples,2);
N = length(samples)/5;
%N = length(samples);

number_points = length(samples);
SIGMA = sample_covariance(samples);

% Bandwidth Calculation
epsilon = 1e-6;
[H,c] = cross_validation_V1(samples,SIGMA);
H = H + epsilon * eye(size(H));


[eval_points, points] = permut(samples,number_points,D,N);
pdf = zeros(length(eval_points),1);
normal_factor = 1 / ((2 * pi)^(D / 2) * sqrt(det(H)));

for i = 1:length(eval_points)

    diff = samples - eval_points(i,:);
    pdf(i) = sum(exp(-0.5 * sum((diff / H) .* diff, 2)));

end

pdf = normal_factor*pdf/N;

if D > 1
    % Reshape to form PDF Matrix of D Dimensions
    reshape_dimension_vector = ones(1,D)*number_points;
    pdf = reshape(pdf,reshape_dimension_vector);
    pdf = pdf/sum(pdf(:));

    evaluation_points = {};

    for i = 1:D

        evaluation_points{i} = reshape(eval_points(:,i),reshape_dimension_vector);

    end
end
evaluation_points = points;
end

%% Evaluation Points by Permutations for KDE

function [evaluation_points, points] = permut(samples,number_points,D,N)

points = cell(1,D);

for i = 1:D
    max_point = max(samples(:,i));
    min_point = min(samples(:,i));
    points{i} = linspace(min_point,max_point,number_points);
end

[grid_mesh{1:D}] = ndgrid(points{:});
evaluation_points = cell2mat(cellfun(@(x) x(:), grid_mesh, 'UniformOutput', false));

end