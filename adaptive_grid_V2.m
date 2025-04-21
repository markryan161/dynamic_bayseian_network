function evaluation_points = adaptive_grid_V2(data, k, gridSize)
% adaptive_grid_V2 computes an adaptive grid over the data space.
% It uses Euclidean distance only and does not output any plots.
%
% INPUTS:
%   data     - an N-by-D matrix of data points
%   k        - number of clusters for K-Means clustering
%   gridSize - scalar defining the number of grid points along each dimension
%
% OUTPUT:
%   evaluation_points - a 1-by-k cell array, where each cell contains 
%                       a subset of grid points (adaptive to cluster density)

dim = size(data, 2);

% K-Means Clustering (assuming myKMeansMulti is available)
[idx, C] = myKMeansMulti(data, k, 20); % K-Means with k clusters

% Global Grid over the data space
gridLimits = [min(data); max(data)]; % Each row is the min and max for each dimension
gridPoints = createGrid(gridLimits, gridSize, dim); % Create a global grid

% Compute Euclidean distances between each grid point and each centroid
distances = computeDistance(gridPoints, C);
[~, clusterAssignment] = min(distances, [], 2);

% Compute the area (proportion of grid points) in each cluster partition
clusterAreas = zeros(k, 1);
for i = 1:k
    clusterAreas(i) = sum(clusterAssignment == i);
end
clusterAreas = clusterAreas / sum(clusterAreas);

% Define adaptive grid density parameters
maxDensity = 5000; 
minDensity = 20;   
gridResolutions = round(minDensity + (maxDensity - minDensity) * (1 - clusterAreas));

% Generate evaluation points in each cluster partition using adaptive density
evaluation_points = [];
all_eval_points = [];  % Initialize an empty matrix for concatenation

for i = 1:k
    clusterMask = (clusterAssignment == i);
    clusterData = gridPoints(clusterMask, :);
    
    stepSize = round(size(clusterData, 1) / gridResolutions(i));
    stepSize = max(stepSize, 1);  % Ensure at least one point is selected
    
    all_eval_points = [all_eval_points; clusterData(1:stepSize:end, :)];
end
evaluation_points = all_eval_points;
end

%% Grid
function gridPoints = createGrid(gridLimits, gridSize, dim)
% createGrid creates a grid of points over the provided limits.
%
% INPUTS:
%   gridLimits - a 2-by-D matrix where first row is the minimum and second row is the maximum for each dimension
%   gridSize   - number of grid points per dimension
%   dim        - number of dimensions
%
% OUTPUT:
%   gridPoints - an (gridSize^dim)-by-D matrix of grid points

ranges = arrayfun(@(i) linspace(gridLimits(1, i), gridLimits(2, i), gridSize), 1:dim, 'UniformOutput', false);
[gridMesh{1:dim}] = ndgrid(ranges{:});
gridPoints = cell2mat(cellfun(@(x) x(:), gridMesh, 'UniformOutput', false));
end

%% Compute Distance
function distances = computeDistance(points, centroids)
% computeDistance calculates the Euclidean distance between each point and each centroid.
%
% INPUTS:
%   points    - an N-by-D matrix of points
%   centroids - a k-by-D matrix of centroids
%
% OUTPUT:
%   distances - an N-by-k matrix where each entry (i,j) is the Euclidean distance 
%               between points(i,:) and centroids(j,:)

% Vectorized Euclidean distance computation:
%   distances(i,j) = sqrt(sum((points(i,:) - centroids(j,:)).^2))
p2 = sum(points.^2, 2);            % N-by-1
c2 = sum(centroids.^2, 2)';          % 1-by-k
%distances = sqrt(bsxfun(@plus, p2, c2) - 2*(points * centroids'));
distances = sqrt(p2+c2- 2*(points * centroids'));
end
