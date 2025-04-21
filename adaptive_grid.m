function evaluation_points = adaptive_grid(data, k, gridSize, distanceMetric)
dim = size(data,2);

    % K-Means Clustering
    [idx, C] = myKMeansMulti(data, k, 20); % K-Means with k clusters

    % Global Grid
    gridLimits = [min(data); max(data)]; % Limits in matrix form
    gridPoints = createGrid(gridLimits, gridSize, dim); % Create grid dynamically

    % distance
    distances = computeDistance(gridPoints, C, distanceMetric);
    [~, clusterAssignment] = min(distances, [], 2);

    clusterAreas = zeros(k, 1);
    for i = 1:k
        clusterAreas(i) = sum(clusterAssignment == i);
    end
    clusterAreas = clusterAreas / sum(clusterAreas); 

    % Adaptive Grid Density
    maxDensity = 5000; 
    minDensity = 20;   
    gridResolutions = round(minDensity + (maxDensity - minDensity) * (1 - clusterAreas));

    % 2D Case
    if dim == 2
        figure; hold on;
        contourf(reshape(gridPoints(:,1), gridSize, gridSize), ...
                 reshape(gridPoints(:,2), gridSize, gridSize), ...
                 reshape(clusterAssignment, gridSize, gridSize), ...
                 numel(unique(idx)), 'LineColor', 'none');
        scatter(samples(:,1), samples(:,2), 10, idx, 'filled', 'k'); % Data points
        scatter(C(:,1), C(:,2), 40, 'k', 'filled'); % Centroids
    end

    %Overlay Adaptive Grids in Each Partition
    step_sizes = zeros(k,1);
    point_count = zeros(k,1);
    evaluation_points = cell(2,k);
    
    for i = 1:k
        clusterMask = (clusterAssignment == i);
        clusterData = gridPoints(clusterMask, :);
        
        stepSize = round(size(clusterData, 1) / gridResolutions(i));
        stepSize = max(stepSize, 1);
        
        if dim == 2
            scatter(clusterData(1:stepSize:end,1), clusterData(1:stepSize:end,2), 5, 'k', 'filled');
        end
        
        step_sizes(i) = stepSize;
        evaluation_points{1,i} = clusterData(1:stepSize:end, :);
        evaluation_points{2,i} = stepSize;
    end


    if dim == 2
        title(['Adaptive Grid Density - ', distanceMetric, ' Distance']);
        xlabel('X'); ylabel('Y');
        grid on; hold off;
    end


end
%%

function gridPoints = createGrid(gridLimits, gridSize, dim)
    ranges = arrayfun(@(i) linspace(gridLimits(1,i), gridLimits(2,i), gridSize), 1:dim, 'UniformOutput', false);
    [gridMesh{1:dim}] = ndgrid(ranges{:});
    gridPoints = cell2mat(cellfun(@(x) x(:), gridMesh, 'UniformOutput', false));
end

%%
function distances = computeDistance(points, centroids, metric)
    if strcmp(metric, 'manhattan')
        distances = sum(abs(points - permute(centroids, [3,2,1])), 2);
    elseif strcmp(metric, 'euclidean')
        distances = sqrt(sum((points - permute(centroids, [3,2,1])).^2, 2));
    else
        error('Invalid distance metric. Use "manhattan" or "euclidean".');
    end
    distances = squeeze(distances);
end
