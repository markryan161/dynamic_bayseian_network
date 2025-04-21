function [likelihood_pdf, evaluation_points] = reweight_V2(particles, observation, D, observation_noise)
% Reweight Function -  Compute the likelihood of the particles given a noisy
% observation i.e. Posterior = Likelihood*Prior 
% Observation = [O1, O2, O3 ...]
% Particles = [X1, X2, X3 ...]

likelihood_particles = zeros(length(particles),1);

R_matrix = eye(D) * observation_noise;
normal_factor = 1/((2*pi)^(D/2)*sqrt(det(R_matrix)));
number_points = length(particles);


%% Sample Permutations
[eval_points, points] = permut(particles,length(particles),D);

for i= 1:length(eval_points)

    diff = observation - eval_points(i,:);
    likelihood_particles(i) = (exp((-1/2) * diff * inv(R_matrix) * diff')); 
    % R Matrix is not inverted here since its uniform

    if likelihood_particles(i) < 

end

likelihood_pdf = likelihood_particles*normal_factor;

if D > 1
    % Reshape to form PDF Matrix of D Dimensions
    reshape_dimension_vector = ones(1,D)*number_points;
    likelihood_pdf = reshape(likelihood_pdf,reshape_dimension_vector);
    likelihood_pdf = likelihood_pdf/sum(likelihood_pdf(:));

    evaluation_points = {};

    for i = 1:D

        evaluation_points{i} = reshape(eval_points(:,i),reshape_dimension_vector);

    end
end
evaluation_points = points;
end


