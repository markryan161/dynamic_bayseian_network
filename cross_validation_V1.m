function [bandwidth,c] = cross_validation_V1(samples, sigma)
%
% LOO LSCV - Leave One Out Least Squares Cross Validation
% H = c*Sigma where c = smoothness factor (scalar) and Sigma = Covariance
% matrix

smoothing_factors = [0.1,1,5,10,50,100,500,1000,5000,10000];
n = size(samples,1);
d = size(samples,2);
LS_total = zeros(length(smoothing_factors),1);

for i = 1:length(smoothing_factors)

    H = smoothing_factors(i)*sigma;
    KDE = multi_KDE_V2(samples,samples,H);
    loo_sum = 0;

    for j = 1:length(samples)
    
        loo_samples = [samples(1:j-1,:);samples(j+1:length(samples),:)];
        loo_pdf = multi_KDE_V2(loo_samples,samples(j),H);
        loo_sum = loo_sum + loo_pdf;

    end

    int_KDE = trapz(KDE.^2);
    LS_total(i) = int_KDE - (2/n)*loo_sum;

end
[~,best_c_id] = min(LS_total);
c= smoothing_factors(best_c_id);
bandwidth = smoothing_factors(best_c_id)*sigma;

end