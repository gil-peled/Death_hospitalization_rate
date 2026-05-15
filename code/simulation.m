%% Appendix A table: E[D_i], E[N_i/c], Corr(lambda_i, D_i), E[H_G'], E[H_G]
% Model from Appendix A:

rng(1245);
%
%   lambda_i ~ Lognormal( log(0.075) - 0.5^2/2 , 0.5^2 )
%   n_it = floor(c * lambda_i) if d_it = 1
%   n_it = 0 otherwise
%
% Enrollment / death mechanism:
%   d_i1 = 1
%   P(d_i,t+1 = 1 | d_it = 0) = 0
%   P(d_i,t+1 = 1 | d_it = 1, n_it = 0) = 1
%   P(d_i,t+1 = 1 | d_it = 1, n_it >= 1) = (1 - gamma*0.0817/c)^(n_it)
%
% Table columns:
%   gamma, E[D_i], E[N_i/c], Corr(lambda_i, D_i), E[H_G'], E[H_G]

clear; clc;

%% Parameters from appendix
c = 1e6;                % nuisance scaling factor
T = 120;                % 120 months = 10 years
gamma_grid = [0, 0.5, 1, 2, 4];
mort30 = 0.0817;        % 30-day post-hospitalization mortality parameter
sigma_lambda = 0.5;     % appendix uses tau = 0.5
mu_lambda = log(0.075) - sigma_lambda^2 / 2;  % ensures E[lambda] = 0.075

%% Lognormal pdf for lambda
f_lambda = @(lam) (1 ./ (lam .* sigma_lambda .* sqrt(2*pi))) .* ...
    exp(- (log(lam) - mu_lambda).^2 ./ (2 * sigma_lambda^2));

%% Moments of lambda
E_lambda = exp(mu_lambda + sigma_lambda^2 / 2);   % = 0.075
E_lambda2 = exp(2*mu_lambda + 2*sigma_lambda^2);
Var_lambda = E_lambda2 - E_lambda^2;

%% Helper handles
ED_cond  = @(lam, gamma) ED_conditional(lam, gamma, c, T, mort30);
ED2_cond = @(lam, gamma) ED2_conditional(lam, gamma, c, T, mort30);
ENc_cond = @(lam, gamma) (floor(c * lam) ./ c) .* ED_cond(lam, gamma);

%% Storage
nG = numel(gamma_grid);
E_D = zeros(nG,1);
E_N_over_c = zeros(nG,1);
Cov_lambda_D = zeros(nG,1);
Corr_lambda_D = zeros(nG,1);
E_HGp = zeros(nG,1);
E_HG = E_lambda * ones(nG,1);

%% Numerical integration options
relTol = 1e-9;
absTol = 1e-12;

%% Loop over gamma values for table
for j = 1:nG
    gamma = gamma_grid(j);

    % E[D_i]
    E_D(j) = integral(@(lam) ED_cond(lam, gamma) .* f_lambda(lam), ...
        0, Inf, 'ArrayValued', true, 'RelTol', relTol, 'AbsTol', absTol);

    % E[N_i/c]
    E_N_over_c(j) = integral(@(lam) ENc_cond(lam, gamma) .* f_lambda(lam), ...
        0, Inf, 'ArrayValued', true, 'RelTol', relTol, 'AbsTol', absTol);

    % E[D_i^2]
    E_D2 = integral(@(lam) ED2_cond(lam, gamma) .* f_lambda(lam), ...
        0, Inf, 'ArrayValued', true, 'RelTol', relTol, 'AbsTol', absTol);

    Var_D = E_D2 - E_D(j)^2;

    % Cov(lambda_i, D_i)
    Cov_lambda_D(j) = E_N_over_c(j) - E_lambda * E_D(j);

    % Corr(lambda_i, D_i)
    denom_table = sqrt(Var_lambda * Var_D);
    if denom_table <= 0 || ~isfinite(denom_table)
        Corr_lambda_D(j) = NaN;
    else
        Corr_lambda_D(j) = Cov_lambda_D(j) / denom_table;
    end

    % E[H_G']
    E_HGp(j) = E_N_over_c(j) / E_D(j);
end

%% Appendix-scale table
results_appendix = table( ...
    gamma_grid(:), E_D, E_N_over_c, Cov_lambda_D, Corr_lambda_D, E_HGp, E_HG, ...
    'VariableNames', {'gamma', 'E_Di', 'E_Ni_over_c', 'Cov_lambda_D', ...
                      'Corr_lambda_D', 'E_HG_prime', 'E_HG'});

disp('Appendix A scale (patient-month units):');
disp(results_appendix);

%% Main-text scaled version
results_maintext = table( ...
    gamma_grid(:), ...
    E_D / 12, ...
    12 * E_N_over_c, ...
    Cov_lambda_D, ...
    Corr_lambda_D, ...
    12 * E_HGp, ...
    12 * E_HG, ...
    'VariableNames', {'gamma', 'E_Di_in_years', 'E_Ni_over_c_per_year', ...
                      'Cov_lambda_D', 'Corr_lambda_D', ...
                      'E_HG_prime_per_year', 'E_HG_per_year'});

disp('Main text scaled version:');
disp(results_maintext);

%% ============================================================
%% Graph Corr(D_i(T), lambda_i) as a function of time in patient-years
%% X-axis = T/12, horizon up to 20 years = 240 months
%% ============================================================

p = 0.0817;
T_grid_months = 1:240;                  % 240 months = 20 years
T_grid_years  = T_grid_months / 12;     % display in patient-years
gamma_grid_plot = [0, 0.5, 1, 2, 4];

Corr_T = NaN(length(T_grid_months), length(gamma_grid_plot));

for j = 1:length(gamma_grid_plot)
    gamma = gamma_grid_plot(j);

    for tt = 1:length(T_grid_months)
        Tcur = T_grid_months(tt);

        % At T=1 month, D_i(T)=1 for everyone, so variance is 0.
        if gamma == 0
            if Tcur == 1
                Corr_T(tt,j) = NaN;
            else
                Corr_T(tt,j) = 0;
            end
            continue;
        end

        % E[D_i(T) | lambda]
        EDT_cond = @(lam) (1 - exp(-gamma * p .* lam * Tcur)) ./ ...
                          (1 - exp(-gamma * p .* lam));

        % E[D_i(T)]
        EDT = integral(@(lam) EDT_cond(lam) .* f_lambda(lam), ...
            0, Inf, 'ArrayValued', true, ...
            'RelTol', relTol, 'AbsTol', absTol);

        % E[lambda_i D_i(T)]
        ElamDT = integral(@(lam) lam .* EDT_cond(lam) .* f_lambda(lam), ...
            0, Inf, 'ArrayValued', true, ...
            'RelTol', relTol, 'AbsTol', absTol);

        % E[D_i(T)^2]
        EDT2 = integral(@(lam) EDT2_continuous_conditional(lam, gamma, p, Tcur) .* f_lambda(lam), ...
            0, Inf, 'ArrayValued', true, ...
            'RelTol', relTol, 'AbsTol', absTol);

        VarDT = EDT2 - EDT^2;
        Cov_lam_DT = ElamDT - E_lambda * EDT;

        denom = sqrt(Var_lambda * VarDT);

        if denom <= 0 || ~isfinite(denom)
            Corr_T(tt,j) = NaN;
        else
            Corr_T(tt,j) = Cov_lam_DT / denom;
        end
    end
end

%% ============================================================
%% Quadratic approximation for gamma = 1 with Corr(0)=0
%% y ≈ b1*T + b2*T^2
%% ============================================================

idx_gamma1 = find(gamma_grid_plot == 1);

% Add origin explicitly
x_all = [0; T_grid_years(:)];
y_all = [0; Corr_T(:, idx_gamma1)];

valid = isfinite(x_all) & isfinite(y_all);
x = x_all(valid);
y = y_all(valid);

% Quadratic through the origin
X_quad = [x, x.^2];
b_quad = X_quad \ y;
yhat_quad = X_quad * b_quad;

% Fitted values on full plotting grid including origin
x_plot = [0; T_grid_years(:)];
yhat_quad_plot = [x_plot, x_plot.^2] * b_quad;

%% ============================================================
%% Plot correlation curves + quadratic approximation
%% ============================================================

figure;
hold on;

for j = 1:length(gamma_grid_plot)
    plot(T_grid_years, Corr_T(:,j), 'LineWidth', 2, ...
        'DisplayName', ['\gamma = ' num2str(gamma_grid_plot(j))]);
end

plot(x_plot, yhat_quad_plot, '--', ...
    'LineWidth', 3, ...
    'DisplayName', '\gamma = 1 quadratic fit');

hold off;

xlabel('Time (patient-years)');
ylabel('Corr(D_i(T), \lambda_i)');
title('Correlation between D_i(T) and \lambda_i');
legend('Location', 'best');
grid on;
xlim([0, 20]);

%% Optional table of plotted values
corr_table = array2table(Corr_T, ...
    'VariableNames', {'gamma_0','gamma_0p5','gamma_1','gamma_2','gamma_4'});
corr_table.T_months = T_grid_months(:);
corr_table.T_years  = T_grid_years(:);
corr_table = movevars(corr_table, {'T_years','T_months'}, 'Before', 1);

disp('Correlation values by time:');
disp(corr_table);

%% Report quadratic coefficients
fprintf('\nGamma = 1 quadratic approximation with Corr(0)=0:\n');
fprintf('Corr(T) ≈ %.6f * T_years %+ .6f * T_years^2\n', ...
    b_quad(1), b_quad(2));

SS_res_quad = sum((y - yhat_quad).^2);
SS_tot = sum((y - mean(y)).^2);
R2_quad = 1 - SS_res_quad / SS_tot;

fprintf('R^2 quadratic-through-origin = %.6f\n', R2_quad);

%% ============================================================
%% Local functions must be at the very end of the script
%% ============================================================

function out = ED_conditional(lam, gamma, c, T, mort30)
    q = (1 - gamma * mort30 / c) .^ floor(c * lam);

    out = zeros(size(q));
    idx1 = abs(q - 1) < 1e-14;
    idx2 = ~idx1;

    out(idx1) = T;
    out(idx2) = (1 - q(idx2).^T) ./ (1 - q(idx2));
end

function out = ED2_conditional(lam, gamma, c, T, mort30)
    q = (1 - gamma * mort30 / c) .^ floor(c * lam);

    out = zeros(size(q));
    idx1 = abs(q - 1) < 1e-14;
    idx2 = ~idx1;

    out(idx1) = T^2;

    if any(idx2)
        qq = q(idx2);

        % S0 = sum_{k=0}^{T-1} q^k
        S0 = (1 - qq.^T) ./ (1 - qq);

        % S1 = sum_{k=0}^{T-1} k q^k
        S1 = qq .* (1 - T * qq.^(T-1) + (T-1) * qq.^T) ./ (1 - qq).^2;

        % sum_{k=0}^{T-1} (2k+1) q^k = 2*S1 + S0
        out(idx2) = 2 * S1 + S0;
    end
end

function out = EDT2_continuous_conditional(lam, gamma, p, Tcur)
    q = exp(-gamma * p .* lam);

    out = zeros(size(q));
    idx1 = abs(q - 1) < 1e-14;
    idx2 = ~idx1;

    out(idx1) = Tcur^2;

    if any(idx2)
        qq = q(idx2);

        % S0 = sum_{k=0}^{Tcur-1} q^k
        S0 = (1 - qq.^Tcur) ./ (1 - qq);

        % S1 = sum_{k=0}^{Tcur-1} k q^k
        S1 = qq .* (1 - Tcur * qq.^(Tcur-1) + (Tcur-1) * qq.^Tcur) ./ (1 - qq).^2;

        out(idx2) = 2 * S1 + S0;
    end
end
