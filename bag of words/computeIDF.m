%load('visionRandom.mat')
load('visionHarris.mat')
[T, n]=size(trainFeatures); 
% allocate space for document idf's
idf = zeros(n, 1); 
% for every document
for j=1:n
    % count non-zero frequency words
    nz = nnz(trainFeatures(:, j));
    % if not zero, assign a weight:
    if nz
        idf(j) = log( T / nz );
    end
end
save idf_Harris.mat idf