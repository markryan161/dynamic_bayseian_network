%% ESS Test - BN

%% Data 1
%charge_particles = BN_test1.ChargeParticles;
%temp_particles = BN_test1.TempParticles;


%% Data 2
charge_particles = CN_test2.ChargeParticles;
temp_particles = CN_test2.TempParticles;
voltage_particles = CN_test2.VoltageParticles;
time_particles = CN_test2.TimeParticles;

x = 0:25:(size(charge_particles,2)-1)*25;
ESS = zeros(1,size(charge_particles,2));

for i = 1:size(charge_particles,2)

   % data1 = [charge_particles(:,i),temp_particles(:,i)];
    data2 = [charge_particles(:,i),temp_particles(:,i),voltage_particles(:,i),time_particles(:,i)];

    ESS(:,i) = kde_ess(data2, 1);

end

plot(x,ESS)
hold on
title('ESS of the Complex Network','Interpreter','latex',FontSize=25)
xlabel('Cycle','Interpreter','latex',FontSize=20)
ylabel('ESS','Interpreter','latex',FontSize=20)
%%%

function degeneracyScore = clusteringDegeneracyTest(particles, numClusters)
% particles: NxD matrix (N particles, D dimensions)
% numClusters: number of clusters to try (e.g., 5)

    % Run k-means clustering
    [clusterIdx, ~] = kmeans(particles, numClusters, 'Replicates', 5);

    % Count how many particles in each cluster
    clusterCounts = histcounts(clusterIdx, 1:(numClusters+1));
    clusterFractions = clusterCounts / sum(clusterCounts);

    % Compute entropy as a proxy for diversity
    entropyVal = -sum(clusterFractions .* log(clusterFractions + eps));

    % Normalize entropy by log of number of clusters (max entropy)
    degeneracyScore = entropyVal / log(numClusters);

end

function ESS = kde_ess(particles, bandwidth)
% KDE-based ESS estimation from particles
% INPUTS:
%   particles : NxD matrix (N particles, D dimensions)
%   bandwidth : scalar or 1xD vector (KDE bandwidth)
% OUTPUT:
%   ESS       : Effective Sample Size based on KDE-derived weights

    [N, D] = size(particles);

    % Estimate density using Gaussian KDE at each particle location
    pdf_vals = zeros(N, 1);
    for i = 1:N
        diffs = particles - particles(i, :);
        kernel_vals = exp(-0.5 * sum((diffs ./ bandwidth).^2, 2));
        pdf_vals(i) = sum(kernel_vals);
    end

    % Normalize KDE output to get pseudo-weights
    weights = pdf_vals / sum(pdf_vals);

    % Compute ESS
    ESS = 1 / sum(weights.^2);
end

