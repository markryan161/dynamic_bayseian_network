function conditional_particles = pdf_to_samples(pdf, number_of_particles, evaluation_points)
% Function to turn pdf to sampled particles
% Sum of particles must be normalized prior to sampleing

pdf_sum = sum(pdf(:));
pmf = pdf/pdf_sum;

conditional_indices = randsample(number_of_particles,number_of_particles,true,pmf);
conditional_particles = evaluation_points(conditional_indices)';

end

