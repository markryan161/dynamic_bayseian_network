function likelihood_pdf = reweight_V3(particles, observation, observation_noise)
% Reweight Function -  Compute the likelihood of the particles given a noisy
% observation i.e. Posterior = Likelihood*Prior 
% For network V1 - observations as a single column vector and same for
% particles
% Observation = [O]
% Particles = [X]

likelihood_particles = zeros(length(particles),1);
observation_variance = observation_noise;
normal_factor = 1/sqrt(2*pi*observation_variance);

for i= 1:length(particles)

    diff = observation - particles(i);
    likelihood_particles(i) = normal_factor * exp(- (diff)^2 / (2 * observation_variance));

    if likelihood_particles(i) < 10^(-100)
        likelihood_particles(i) = 10^(-100);
    end
    
end

likelihood_pdf = likelihood_particles;
