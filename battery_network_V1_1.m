%% V1 Battery Network
% Using maximum battery temperature as a variable with battery degradation
% (%). KDE used to form probability distributions. Assuming temperature
% measurements have noise variance (sigma) of 0.005 deg. Same for charges
% of 0.005 units variance.
% HOW DO WE TRANSITION PARTICLES???? BIG QUESTION - USING MOVING AVERAGES
% FOR THIS EXAMPLE - STARTING WITH DEGRADATION AND TEMPERATURE BEING
% PROPOGATED WITH MEAN OF ZERO AND VARIANCE OF 0.05 (10X MEASUREMENT NOISE)
% V2 - EXPAND NETWORK AND USE REGRESSION FOR DATA
% DRIVEN TRANSITION MODEL
% Step 1 - Overview - Have Prediction step done for gold - adapt for this
% example. Need KDE 4 times so make its own m-file. A
% Every 25 cycles - state estimate

%% Initializing Observation Data
mod(4,4)
max_temps_observations = max_temps;
max_charge_observations = max_charge;

%% Number of Cycles
total_cycles = 4*length(max_temps_observations);

%% BUG FIXING
max_likelihood_pdf_store = zeros(1,total_cycles);

%% Variance
temp_particle_variance = 0.05;
charge_particle_variance = 4;

temp_observation_noise = 0.005;
charge_observation_noise = 0.005;


%% Particles
particles = 50;
proposal_particles = ones(particles,1);

%% Initialize Particle Stores
temp_particle_store = zeros(particles,total_cycles);
charge_particle_store = temp_particle_store;
temp_pdf_store = temp_particle_store;
charge_pdf_store = temp_particle_store;
smoothing_factor_store = zeros(4,total_cycles);

%% Particle Filter

for i = 1:total_cycles
    
    %% Predict - KDE for P(Temp_t|Deg_t-1) and KDE for P(Deg_t|Temp_t)
    % At t = 1, apply measurement noise - if not KDE wont work for t = 2
    if i == 1
        temp_proposal_particles = proposal_particles*max_temps_observations(i);
        charge_proposal_particles = proposal_particles*max_charge_observations(i);

    else
        
        % P(Temp_t|Deg_t-1)
        temp_proposal_particles = temp_proposal_particles + temp_particle_variance * randn(particles,1);
        samples = [temp_proposal_particles, charge_proposal_particles];

        [joint_pdf, joint_evaluation_points,c] = multi_KDE_V3(samples);
        smoothing_factor_store(1,i) = c;

        [marginal_pdf, ~] = multi_KDE_V3(charge_proposal_particles);
        smoothing_factor_store(2,i) = c;

        conditional_temp_pdf = joint_pdf/marginal_pdf';
        conditional_temp_pdf = conditional_temp_pdf/sum(conditional_temp_pdf);

        % Sample P(Temp_t|Deg_t-1)
        points = joint_evaluation_points{1};
        conditional_temp_particles = pdf_to_samples(conditional_temp_pdf,particles,points);

        % P(Deg_t|Temp_t)
        charge_proposal_particles = charge_proposal_particles + charge_particle_variance * randn(particles,1);
        samples = [charge_proposal_particles, conditional_temp_particles];
        [joint_pdf,joint_evaluation_points,c] = multi_KDE_V3(samples);
        smoothing_factor_store(3,i) = c;

        [marginal_pdf, marginal_evaluation_points,c] = multi_KDE_V3(conditional_temp_particles);
        smoothing_factor_store(4,i) = c;
        
        conditional_charge_pdf = joint_pdf/marginal_pdf';
        conditional_charge_pdf = conditional_charge_pdf/sum(conditional_charge_pdf);

        % Sample P(Deg_t|Temp_t)
        points = joint_evaluation_points{1};
        conditional_charge_particles = pdf_to_samples(conditional_charge_pdf,length(conditional_charge_pdf),points);
        
        %% Observe and Reweight - P(Temp_t | O_Temp_t) ~ P(O_Temp_t | Temp_t) * P(Temp_t) and P(Deg_t | O_Deg_t) ~ P(O_Deg_t | Deg_t) * P(Deg_t)
        if mod(i-1,4) == 0

            observation_count = (i-1)/4
            temp_observation  = max_temps_observations(observation_count,1);
            charge_observation = max_charge_observations(observation_count,1);

            temp_likelihood_pdf= reweight_V3(conditional_temp_particles,temp_observation,temp_observation_noise);
            temp_likelihood_pdf = temp_likelihood_pdf/sum(temp_likelihood_pdf);

            charge_likelihood_pdf = reweight_V3(conditional_charge_particles,charge_observation,charge_observation_noise);
            charge_likelihood_pdf = charge_likelihood_pdf/sum(charge_likelihood_pdf);

            posterior_temp_pdf = temp_likelihood_pdf .* conditional_temp_pdf;
            posterior_charge_pdf = charge_likelihood_pdf .* conditional_charge_pdf;

            reweighted_temp_particles  = univariate_sample_particles(conditional_temp_particles,posterior_temp_pdf)';
            reweighted_charge_particles = univariate_sample_particles(conditional_charge_particles,posterior_charge_pdf)';

            temp_proposal_particles = reweighted_temp_particles;
            charge_proposal_particles = reweighted_charge_particles;

            temp_pdf_store(:,i) = temp_likelihood_pdf;
            charge_pdf_store(:,i) = charge_likelihood_pdf;

            % Need to calculate prediction intervals for likelihood
            
        else

            temp_proposal_particles = conditional_temp_particles;
            charge_proposal_particles = conditional_charge_particles;

            temp_pdf_store(:,i) = conditional_temp_pdf;
            charge_pdf_store(:,i) = conditional_charge_pdf;

        end % Observation IF


    end % Cycle 1 IF
    temp_particle_store(:,i) = temp_proposal_particles;
    charge_particle_store(:,i) = charge_proposal_particles;

end % Main FOR

%% State Estimates
temp_means = zeros(total_cycles,1);
charge_means = zeros(total_cycles,1);
temp_cl_upper = zeros(total_cycles,1);
temp_cl_lower = temp_cl_upper;
charge_cl_upper = temp_cl_upper;
charge_cl_lower = temp_cl_upper;
temp_actual = temp_means;
charge_actual = temp_means;

for i = 1:size(temp_particle_store,2)

    temp_means(i) = mean(temp_particle_store(:,i));
    charge_means(i) = mean(charge_particle_store(:,i));

end

max_temps_observations(16) = 41;

clf;
subplot(2,2,1)
plot(1:total_cycles,charge_means)

subplot(2,2,2)
plot(1:length(max_charge_observations),max_charge_observations)

subplot(2,2,3)
plot(1:total_cycles,temp_means)

subplot(2,2,4)
plot(1:length(max_temps_observations),max_temps_observations)