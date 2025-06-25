% Example signal (e.g., a noisy sine wave)
N = 1024;
t = linspace(0, 1, N);
signal = sin(2*pi*10*t) + 0.3*randn(1, N);  % Signal with noise

% Choose a wavelet, for example 'db1' (Daubechies 1)
waveletName = 'db1';

% Try different decomposition levels
maxLevel = floor(log2(N));
mse = zeros(1, maxLevel);

for J = 1:maxLevel
    % Perform MODWT at level J
    wt = modwt(signal, waveletName, J);  % Only the wavelet coefficients are needed
    
    % Reconstruct the signal from the wavelet coefficients
    reconstructedSignal = imodwt(wt, waveletName);
    
    % Calculate the Mean Squared Error (MSE) between original and reconstructed signal
    mse(J) = mean((signal - reconstructedSignal).^2);
end

% Plot MSE to identify the optimal level
figure;
plot(1:maxLevel, mse, '-o');
xlabel('Decomposition Level');
ylabel('Mean Squared Error');
title('Reconstruction Error (MSE) vs. Decomposition Level');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%
%%%%%%%
%%%%%%%     Entropy calculation example for MODWT coefficients
%%%%%%%
%%%%%%%
%%%%%%% Example signal (e.g., a noisy sine wave)

N = 1024;
t = linspace(0, 1, N);
signal = sin(2*pi*10*t) + 0.3*randn(1, N);  % Signal with noise

% Choose a wavelet, for example 'db1' (Daubechies 1)
waveletName = 'db1';
maxLevel = floor(log2(N));


entropyValues = zeros(1, maxLevel);
for J = 1:maxLevel
    % Perform MODWT at level J
    wt = modwt(signal, waveletName, J);
    
    % Calculate the entropy of the wavelet coefficients at each level
    entropyValues(J) = entropy(wt(:));  % Flatten the coefficients into a vector
end

% Plot entropy to identify the optimal level
figure;
plot(1:maxLevel, entropyValues, '-o');
xlabel('Decomposition Level');
ylabel('Entropy');
title('Entropy vs. Decomposition Level');
