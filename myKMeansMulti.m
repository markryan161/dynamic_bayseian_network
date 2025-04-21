function [idx, C] = myKMeansMulti(X, k, max_iters)
   

    rng(1); % Fix seed for reproducibility
    N = size(X, 1); % Number of data points
    D = size(X, 2); % Number of dimensions (features)

    % Step 1: Randomly Initialize Centroids
    rand_idx = randperm(N, k); % Select k random indices
    C = X(rand_idx, :); % Initial centroids (k x D)

    % Step 2: Iterate Until Convergence or Max Iterations
    idx = zeros(N, 1); % Cluster assignments
    for iter = 1:max_iters
        % Step 2a: Assign each point to the closest centroid
        for i = 1:N
            distances = sum((C - X(i, :)).^2, 2); % Squared Euclidean distance
            [~, idx(i)] = min(distances); % Assign to closest centroid
        end

        % Step 2b: Compute new centroids as the mean of assigned points
        new_C = zeros(k, D);
        for j = 1:k
            cluster_points = X(idx == j, :); % Points assigned to cluster j
            if ~isempty(cluster_points)
                new_C(j, :) = mean(cluster_points, 1); % Compute mean
            else
                % Handle empty clusters by reinitializing
                new_C(j, :) = X(randi(N), :);
            end
        end

        % Step 3: Check for Convergence
        if isequal(C, new_C)
            break;
        end
        C = new_C; % Update centroids
    end
end