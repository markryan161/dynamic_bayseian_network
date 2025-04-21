function pdf = multi_KDE_V2(samples, evaluation_points, H)

%% MULTIVARIATE KDE V2 
% For LSCV 

D = size(samples,2);
N = length(samples);

pdf = zeros(length(evaluation_points),1);
normal_factor = 1 / ((2 * pi)^(D / 2) * sqrt(det(H)));

for i = 1:length(evaluation_points)

    diff = samples - evaluation_points(i,:);
    pdf(i) = sum(exp(-0.5 * sum((diff / H) .* diff, 2)));

end

pdf = normal_factor*pdf/N;

end
