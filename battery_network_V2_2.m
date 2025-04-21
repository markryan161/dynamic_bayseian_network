%% V2 Battery Network - Long Term
% Using maximum battery temperature, maximum_voltage, and charging time as
% parent variables of degadation
% 
% Every 25 cycles - state estimate

%% Initializing Observation Data
mod(4,4)
max_temps_observations = max_temps;
max_charge_observations = max_charge;
max_voltage_observations = max_voltages;
min_charge_observations = min_charge;
min_temp_observations = min_temps;
min_voltage_observations = min_voltages;
time_observations = times(1,:)';

%% Number of Cycles
total_cycles = length(max_temps_observations)+23;

%% BUG FIXING
max_likelihood_pdf_store = zeros(1,total_cycles);

%% Variance
temp_particle_variance = 0.05;
charge_particle_variance = 4;
voltage_particle_variance = 0.008;
time_particle_variance = 100;

temp_observation_noise = 0.005;
charge_observation_noise = 0.005;
voltage_particle_noise = 0.001;
time_particle_noise = 10;



%% Particles
particles = 40;
proposal_particles = ones(particles,1);

%% Initialize Particle Stores
temp_particle_store = zeros(particles,total_cycles);
charge_particle_store = temp_particle_store;
voltage_particle_store = temp_particle_store;
time_particle_store = temp_particle_store;

temp_pdf_store = temp_particle_store;
charge_pdf_store = temp_particle_store;
voltage_pdf_store = time_particle_store;
time_pdf_store = temp_particle_store;

%% Smoothing Factor Store
smoothing_factor_store = zeros(4,total_cycles);

%% Particle Filter



for i = 1:total_cycles

    %% Propose
    % First Cycle

    if i == 1

        temp_proposal_particles = proposal_particles*max_temps_observations(i);
        charge_proposal_particles = proposal_particles*max_charge_observations(i);
        voltage_proposal_particles = proposal_particles*max_voltage_observations(i);
        time_proposal_particles = proposal_particles*time_observations(i);

    else
        
        % Calc. P(Deg_t-1)
        
         % P(Temp_t|Deg_t-1) = P(Temp_t, Deg_t-1)/P(Deg_t-1)
         temp_proposal_particles = temp_proposal_particles + temp_particle_variance * randn(particles,1);
         samples = [temp_proposal_particles,charge_proposal_particles];
         [joint_pdf, joint_evaluation_points] = multi_KDE(samples);
        % smoothing_factor_store(1,i) = c;
         samples = [charge_proposal_particles];
         [marginal_pdf, ~] = multi_KDE(samples);
        % smoothing_factor_store(2,i) = c;
         conditional_temp_pdf = joint_pdf/marginal_pdf';
         conditional_temp_pdf = conditional_temp_pdf/sum(conditional_temp_pdf);
         points = joint_evaluation_points{1};
         conditional_temp_particles = pdf_to_samples(conditional_temp_pdf,particles,points);

         % P(Voltage_t|Deg_t-1) = P(Voltage_t, Deg_t-1)/P(Deg_t-1)
         voltage_proposal_particles = voltage_proposal_particles + voltage_particle_variance * randn(particles,1);
         samples = [voltage_proposal_particles,charge_proposal_particles];
         [joint_pdf, joint_evaluation_points] = multi_KDE(samples);
        % smoothing_factor_store(3,i) = c;
         samples = [charge_proposal_particles];
         [marginal_pdf, ~] = multi_KDE(samples);
        % smoothing_factor_store(4,i) = c;
         conditional_voltage_pdf = joint_pdf/marginal_pdf';
         conditional_voltage_pdf = conditional_voltage_pdf/sum(conditional_voltage_pdf);
         points = joint_evaluation_points{1};
         conditional_voltage_particles = pdf_to_samples(conditional_temp_pdf,particles,points);

         % P(Time_t|Deg_t-1) = P(Time_t, Deg_t-1)/P(Deg_t-1)
         time_proposal_particles = time_proposal_particles + time_particle_variance* randn(particles,1);
         samples = [time_proposal_particles,charge_proposal_particles];
         [joint_pdf, joint_evaluation_points] = multi_KDE(samples);
        % smoothing_factor_store(5,i) = c;
         samples = [charge_proposal_particles];
         [marginal_pdf, ~] = multi_KDE(samples);
        % smoothing_factor_store(6,i) = c;
         conditional_time_pdf = joint_pdf/marginal_pdf';
         conditional_time_pdf = conditional_voltage_pdf/sum(conditional_time_pdf);
         points = joint_evaluation_points{1};
         conditional_time_particles = pdf_to_samples(conditional_time_pdf,particles,points);

        % P(Deg_t|Temp_t, Voltage_t, Time_t) = P(Deg_t,Temp_t, Voltage_t,
        % Time_t)/P(Temp_t, Voltage_t,Time_t)
        charge_proposal_particles = charge_proposal_particles + charge_particle_variance * randn(particles,1);
        samples = [charge_proposal_particles, conditional_temp_particles, conditional_voltage_particles, conditional_time_particles];
        [joint_pdf,g1] = multi_KDE_V4(samples);
       % smoothing_factor_store(7,i) = c;
        samples = [conditional_temp_particles, conditional_voltage_particles, conditional_time_particles];
        [marginal_pdf, marginal_evaluation_points] = multi_KDE(samples);
        %smoothing_factor_store(4,i) = c;
        conditional_charge_pdf = matrix_inverse(joint_pdf, marginal_pdf);
        conditional_charge_pdf = conditional_charge_pdf/sum(conditional_charge_pdf);
        points = g1;
        conditional_charge_particles = pdf_to_samples(conditional_charge_pdf,length(conditional_charge_pdf),points);
        
        %% Observe and Reweight - P(Temp_t | O_Temp_t) ~ P(O_Temp_t | Temp_t) * P(Temp_t) and P(Deg_t | O_Deg_t) ~ P(O_Deg_t | Deg_t) * P(Deg_t)
        if i <= length(max_temps_observations)

            observation_count = i
            temp_observation  = max_temps_observations(observation_count,1);
            charge_observation = max_charge_observations(observation_count,1);
            voltage_observation = max_voltage_observations(observation_count,1);
            time_observation = time_observations(observation_count,1);

            temp_likelihood_pdf= reweight_V3(conditional_temp_particles,temp_observation,temp_observation_noise);
            temp_likelihood_pdf = temp_likelihood_pdf/sum(temp_likelihood_pdf);

            voltage_likelihood_pdf= reweight_V3(conditional_voltage_particles,voltage_observation,voltage_particle_noise);
            voltage_likelihood_pdf = voltage_likelihood_pdf/sum(voltage_likelihood_pdf);

            time_likelihood_pdf= reweight_V3(conditional_time_particles,time_observation,time_particle_noise);
            time_likelihood_pdf = time_likelihood_pdf/sum(time_likelihood_pdf);

            charge_likelihood_pdf = reweight_V3(conditional_charge_particles,charge_observation,charge_observation_noise);
            charge_likelihood_pdf = charge_likelihood_pdf/sum(charge_likelihood_pdf);

            posterior_temp_pdf = temp_likelihood_pdf .* conditional_temp_pdf;
            posterior_charge_pdf = charge_likelihood_pdf .* conditional_charge_pdf;
            posterior_time_pdf = time_likelihood_pdf .* conditional_time_pdf;
            posterior_voltage_pdf = voltage_likelihood_pdf .* conditional_voltage_pdf;

            reweighted_temp_particles  = univariate_sample_particles(conditional_temp_particles,posterior_temp_pdf)';
            reweighted_charge_particles = univariate_sample_particles(conditional_charge_particles,posterior_charge_pdf)';
            reweighted_voltage_particles = univariate_sample_particles(conditional_voltage_particles,posterior_voltage_pdf)';
            reweighted_time_particles = univariate_sample_particles(conditional_time_particles,posterior_time_pdf)';


            temp_proposal_particles = reweighted_temp_particles;
            charge_proposal_particles = reweighted_charge_particles;
            voltage_proposal_particles = reweighted_voltage_particles;
            time_proposal_particles = reweighted_time_particles;

            temp_pdf_store(:,i) = posterior_temp_pdf;
            charge_pdf_store(:,i) = posterior_charge_pdf;
            voltage_pdf_store(:,i) = posterior_voltage_pdf;
            time_pdf_store(:,i) = posterior_time_pdf;

            % Need to calculate prediction intervals for likelihood
            
        else

            temp_proposal_particles = conditional_temp_particles;
            charge_proposal_particles = conditional_charge_particles;
            voltage_proposal_particles = conditional_voltage_particles;
            time_proposal_particles = conditional_time_particles;

            temp_pdf_store(:,i) = conditional_temp_pdf;
            charge_pdf_store(:,i) = conditional_charge_pdf;
            voltage_pdf_store(:,i) = conditional_voltage_pdf;
            time_pdf_store(:,i) = conditional_time_pdf;

        end % Observation IF

    end % Cycle 1 IF

     temp_particle_store(:,i) = temp_proposal_particles;
     voltage_particle_store(:,i) = voltage_proposal_particles;
     charge_particle_store(:,i) = charge_proposal_particles;
     time_particle_store(:,i) = time_proposal_particles;

end % Main FOR

function conditional_pdf = matrix_inverse(joint, marginal)
    % MATRIX_INVERSE Compute conditional PDF from joint (4D) and marginal (3D)
    % Replaces any NaNs in the input with eps and avoids division by zero.

    % Replace NaNs with eps
    joint(isnan(joint)) = eps;
    marginal(isnan(marginal)) = eps;

    % Extract size
    [dim1, dim2, dim3, dim4] = size(joint);

    % Flatten
    A_matrix = reshape(joint, [], dim4);  % [abc × d]
    b_vector = marginal(:);               % [abc × 1]

    % Avoid divide-by-zero by replacing 0s with eps
    b_vector(b_vector == 0) = eps;

    % Pseudo-inverse division
    result = A_matrix ./ b_vector;        % [abc × d]

    % Normalize to sum to 1
    conditional_pdf = sum(result, 1);     % [1 × d]
    conditional_pdf(isnan(conditional_pdf)) = eps;
    conditional_pdf = conditional_pdf / sum(conditional_pdf);  % normalize
    conditional_pdf = conditional_pdf';   % [d × 1]
end
