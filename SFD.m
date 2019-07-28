
%% calcPSFD_withKernels
% Description
% @author Keara Burke (knburke@orex.lpl.arizona.edu) 2018
%         
% This program employs the revised recommended methods for analyzing
% crater size-frequency distributions. Almost entirely a translation of
% the Igor script written by Stuart Robbins. Any errors are likely my own.
%
% Resources:
% (1) Robbins, S.J. et al. (2018). Revised recommended methods for analyzing
%     crater size-frequency distributions. Meteoritics & Planetary 
%     Science, 1-41.
% (2) Clauset, A., Shalizi CR., and Newman, MEJ. (2009). Power-law
%     distributions in empirical data. SIAM, 51(4), 661-703.
%
%% Version Notes
%  
% July 2018
%  - Does not have handling for sigma_range > 5
%  - Does not have error handling on par with Stuart's yet
%  - Currently uses Clauset script for determining completeness limit
%  - Uses [diff, idx] = min( abs(edf - level) ); as equivalent to find level
%    from igor, does not interpolate, could pose a problem if edf_diam does
%    not have values close enough to test_diam
%  - Uses rand() function to replace enoise from igor
%
% August 2018
%  - Calculates completeness limit using the rollover from the
%    derivative of the ISFD, but has the option of using the Clauset MLE method
%  - Deprecated previous findLevel method in favor of a function more
%    similar to the WaveMetric description of findLevel. This function
%    returns the point number of LevelX and can be combined with interp1()
%    to get values similar to Igor. Added handling for non-monotonic case
%    to perform this step iteratively.
%  - Doesn't have the equation for a uniform kernel because I was being
%    lazy.

%% I. User Modified Parameters

tic % timing funciton

% Change the following as needed
diam            = sort(table2array(data));  % location of measurements
diam_sd         = 0.1 .* diam;              % standard deviation of measurements (use 0.1*diam if only 1 counter)
diam_start      = 0;                        % allows user to exclude data smaller than this value
area            = 7.786*10^5;%490.8739;     % surface area measured on
measureUnit     = 'm';                      % units for diameter measurements (for plotting purposes)
sigFigSpec      = '%5.4f';                  % sig figs (for plotting purposes)
fontSizeSpec    = 16;                       % font size  (for plotting purposes)

% Only change the following if you have read Robbins et al. 2018 (1)
sigma_var       = 0.1;                      % a priori uncertainty factor
discretization  = 0.01;                     % epsilon in eq. 4
sigma_range     = 5;                        % range for every data point
kernel          = 'g';                      % g: gaussian (default), u: uniform, t: triangular, c: cosine, e: epanechnikov
max_monteCarlos = 1000;                     % maximum number of monte carlos to run before giving up on convergence
min_monteCarlos = 100;                      % minimum number of monte carlos to run before checking convergence value
mc_convergence  = 2e-3;                     % threshold for monte carlo convergence
confidence      = 2;                        % units of sigma, this corresponds to 95.54 perc
bs_sigma		= 1;                		% bootstrap samples original diameters if difference between adjacent diameters is < bs_sigma * sum of the adjacent diameter uncertainties
apriori_uncert  = 0.1;                      % apriori uncertainty for diameter measurements
apriori_N_var   = 0.0;                      % apriori in number of craters found as a strict fraction of the total (e.g., "0.3" = ±30%)
complete_level  = 0.8;                      % finds "level" steady-state for first derivative of edf_isfd, completeness is where value is this fraction 

%% II. Variable Initialization

rng('shuffle', 'twister'); % to ensure true randomness. sometimes throws errors so just comment out if needed

if max_monteCarlos < min_monteCarlos
    fprintf('Notice: Max number of Monte Carlo runs is less than min number. Defaulting to %d runs.\n', 2*min_monteCarlos);
    max_monteCarlos = min_monteCarlos;
end

N                   = length(diam);
diam                = sort(diam(diam >= diam_start));
diam_min            = max( diam_start, min(diam) - ( sigma_range * diam_sd(1)) );
diam_max            = max(diam) + ( sigma_range * diam_sd(end) );
num_kde             = 1 + floor( log10( diam_max / diam_min ) / discretization ); % v in eq. 5
edf_dsfd            = zeros(num_kde, 1);
edf_diam            = 10.^(log10(diam_min) + ((1:num_kde)'-1)*discretization);
bootstrap_data      = zeros(num_kde, max_monteCarlos);
bootstrap_conv      = zeros(num_kde, ceil(max_monteCarlos/min_monteCarlos));
bootstrap_conv_dsfd = zeros(num_kde, 1);

edf_erro_dsfd_pos   = zeros(num_kde, 1);
edf_erro_dsfd_neg   = zeros(num_kde, 1);
edf_erro_dsfd_1s_pos   = zeros(num_kde, 1);
edf_erro_dsfd_1s_neg   = zeros(num_kde, 1);

fprintf('\nCalculate Size Frequency Distribution using Kernel Density');
fprintf('\n\nThis code factors in:\n  +/-%2d%%    uncertainty in size diameter',apriori_uncert*100);
fprintf('\n  +/-%2d%%    uncertainty in the number of diameters measured',apriori_N_var*100);
fprintf('\n  %d      Monte Carlo runs for the bootstrap-based uncertainty at %d-sigma (+/-%4.2f%%)\n', max_monteCarlos, confidence, erf(confidence/sqrt(2))*100);

%% III. Bin data

fprintf('\n\nPreparing distributions.');

% DSFD    
for i = 1:N
    % since we use these values to index edf_dsfd, we have to round
    start = round(findLevel(edf_diam, diam(i) - diam_sd(i) * sigma_range, 1));
    stop  = round(findLevel(edf_diam, diam(i) + diam_sd(i) * sigma_range, num_kde));
    
    for j = start:stop
        edf_dsfd(j) = kde_fun(edf_dsfd(j), edf_diam(j), diam(i), diam_sd(i), kernel, N);
    end
end

% RSFD
edf_rsfd = edf_dsfd .* edf_diam.^(3); 

% ISFD
edf_isfd = edf_dsfd;
for i = 1:num_kde-1
    edf_isfd(i) = (edf_dsfd(i) + edf_dsfd(i+1)) / 2 * ( edf_diam(i+1) - edf_diam(i) );
end

% CSFD
dummy = flipud(edf_isfd);
for i = 2:length(dummy)
    dummy(i) = dummy(i-1)+dummy(i);
end

edf_csfd        = flipud(dummy(1:end));
edf_csfd(end) = 0;

edf_isfd(end) = 0; % the zero will get ignored in the end
%% IV. Bootstrap (hybrid method) & Create CIs

fprintf('\n\nBootstrapping. This can be a slow process.');

for mc_run = 1:max_monteCarlos
    
    num_bootstraps = round(N*(-apriori_N_var+rand(1)*(2*apriori_N_var)+1));	% models enoise, so we can vary number of points per bootstrap
    
    for i=1:num_bootstraps
        
        %sample_loc = 1+sample_locs(i);
        % Hybrid between smoothed and direct sampling
        sample_loc  = 1 + rand(1) * N; %create a uniformly distributed random variate
        
        if sample_loc <= 1.5 % first diam
            sample_diam = diam(1);
            
        elseif sample_loc >= length(diam) - 1 % last diam
            
            if sample_loc > length(diam)
                sample_loc = length(diam);
            end
            
            idx = findLevel(edf_diam, interp1(diam, sample_loc));
            sample_diam = interp1(edf_diam, idx);
            
        elseif ( diam(round(sample_loc)) - diam(round(sample_loc)-1) ) < ... % dense region
               ( bs_sigma * apriori_uncert * ( diam(round(sample_loc)) + diam(round(sample_loc)-1)) ) && ...
               ( diam(round(sample_loc)+1) - diam(round(sample_loc)) ) < ...
               ( bs_sigma * apriori_uncert * ( diam(round(sample_loc)+1) + diam(round(sample_loc))) )
           
            sample_diam = diam(round(sample_loc));
            
        else % sparse region
            idx = findLevel(edf_diam, interp1(diam, sample_loc));
            sample_diam = interp1(edf_diam, idx);
            
        end
        
        sample_sd = sample_diam * apriori_uncert;

        % since we use these values to index bootstrap_data, we have to round
        start = round(findLevel(edf_diam, sample_diam - sample_sd * sigma_range, 1));
        stop  = round(findLevel(edf_diam, sample_diam + sample_sd * sigma_range, num_kde));
        
         for j = start:stop
             bootstrap_data(j,mc_run) = kde_fun(bootstrap_data(j,mc_run), edf_diam(j), sample_diam, sample_sd, kernel, N);
         end
         
    end
    
    if mod(mc_run, min_monteCarlos) == 0 % Every min_monteCarlos runs, do this
        dummy = zeros(mc_run,1);
        for i=1:num_kde
            dummy = sort(bootstrap_data(i,1:mc_run))';
            
            idx = findLevel(dummy, edf_dsfd(i), 0);
            
            if idx == 0
                bootstrap_conv(i,mc_run/min_monteCarlos) = 0;
            else
                bootstrap_conv(i,mc_run/min_monteCarlos) = abs( edf_dsfd(i)-interp1(dummy, idx + (mc_run-idx)*erf(confidence/sqrt(2))) );
            end
            
            idx1 = findLevel(dummy, edf_dsfd(i), NaN);
            
            if isnan(idx1)
                edf_erro_dsfd_neg(i)    = 0;
                edf_erro_dsfd_pos(i)    = 0;
                edf_erro_dsfd_1s_neg(i) = edf_dsfd(i);
                edf_erro_dsfd_1s_pos(i) = edf_dsfd(i);
            else
                edf_erro_dsfd_neg(i)    = abs( edf_dsfd(i) - interp1(dummy, idx1 * (1 - erf(confidence/sqrt(2)))) );
                edf_erro_dsfd_pos(i)    = abs( edf_dsfd(i) - interp1(dummy, idx1 + (max_monteCarlos - idx1) * erf(confidence/sqrt(2))) );
                edf_erro_dsfd_1s_neg(i) = interp1(dummy, idx1 * (1 - erf(1/sqrt(2))));
                edf_erro_dsfd_1s_pos(i) = interp1(dummy, idx1 + (max_monteCarlos - idx1) * erf(1/sqrt(2)));
            end
        end
        
        if mc_run / min_monteCarlos >= 2 % After two * min_monteCarlos runs, do this
            
            for i=1:num_kde
                bootstrap_conv_dsfd(i) = abs( bootstrap_conv(i,mc_run/min_monteCarlos) - bootstrap_conv(i,mc_run/min_monteCarlos-1) ) / bootstrap_conv(i,mc_run/min_monteCarlos);
            end
            
            if mean(bootstrap_conv_dsfd) <= mc_convergence % check for convergence
                fprintf('Bootstrap converged to %f after %d iterations.\n', mc_convergence, mc_run);
                max_monteCarlos = mc_run;
            end
        end
    end
    
end

%% V. Scale Data

% Scale the data so our plot axes actually mean something in the end.
scaling = length(diam) / edf_csfd(1) / area;

bootstrap_data          = bootstrap_data .* scaling;
bootstrap_conv          = bootstrap_conv .* scaling;

edf_dsfd                = edf_dsfd .* scaling;
edf_erro_dsfd_neg       = edf_erro_dsfd_neg .* scaling;
edf_erro_dsfd_pos       = edf_erro_dsfd_pos .* scaling;
edf_erro_dsfd_1s_neg    = edf_erro_dsfd_1s_neg .* scaling;
edf_erro_dsfd_1s_pos    = edf_erro_dsfd_1s_pos .* scaling;

edf_rsfd                = edf_rsfd .* scaling;
edf_erro_rsfd_neg		= edf_rsfd - (edf_erro_dsfd_neg ./ edf_dsfd .* edf_rsfd);
edf_erro_rsfd_pos		= edf_rsfd + (edf_erro_dsfd_pos ./ edf_dsfd .* edf_rsfd);
edf_erro_rsfd_1s_neg	= edf_erro_dsfd_1s_neg ./ edf_dsfd .* edf_rsfd;
edf_erro_rsfd_1s_pos	= edf_erro_dsfd_1s_pos ./ edf_dsfd .* edf_rsfd;

edf_isfd                = edf_isfd .* scaling;
edf_erro_isfd_neg		= edf_isfd - (edf_erro_dsfd_neg ./ edf_dsfd .* edf_isfd);
edf_erro_isfd_pos		= edf_isfd + (edf_erro_dsfd_pos ./ edf_dsfd .* edf_isfd);
edf_erro_isfd_1s_neg	= edf_erro_dsfd_1s_neg ./ edf_dsfd .* edf_isfd;
edf_erro_isfd_1s_pos	= edf_erro_dsfd_1s_pos ./ edf_dsfd .* edf_isfd;

edf_csfd                = edf_csfd .* scaling;
edf_erro_csfd_neg		= edf_csfd - (edf_erro_dsfd_neg ./ edf_dsfd .* edf_csfd);
edf_erro_csfd_pos		= edf_csfd + (edf_erro_dsfd_pos ./ edf_dsfd .* edf_csfd);
edf_erro_csfd_1s_neg	= edf_erro_dsfd_1s_neg ./ edf_dsfd .* edf_csfd;
edf_erro_csfd_1s_pos	= edf_erro_dsfd_1s_pos ./ edf_dsfd .* edf_csfd;

edf_erro_dsfd_neg       = edf_dsfd - edf_erro_dsfd_neg;
edf_erro_dsfd_pos       = edf_dsfd + edf_erro_dsfd_pos;      
%% VI. Find Completeness Limit

fprintf('Finding the completeness limit');

% You can run this section in one of two ways, just comment out the other
% 1. Clauset wrote scripts which utilize the maximum likelihood estimate method
% 2. Stuart's method takes the derivative of the incremental plot and finds
%    the rollover point

% Method 1 (you'll need to download this script if you don't have it: 
% http://tuvalu.santafe.edu/~aaronc/powerlaws/)
%[alpha, xmin, L] = plfit( diam );
%completeness = xmin;

% Method 2
edf_isfd_diff = edf_isfd;
edf_isfd_diff(end) = []; % here is where we ignore that zero from before

for i=1:length(edf_isfd)-1
    edf_isfd_diff(i) = log10(edf_isfd(i+1))-log10(edf_isfd(i));
end

edf_isfd_diff(end) = NaN;

[~, max_pts] = max(edf_isfd);
next_zero = max_pts + ceil(findLevel(edf_isfd_diff(max_pts+1:end-1), 0, length(edf_isfd_diff)-1-max_pts));

if next_zero-max_pts <= 10
    fprintf("Difference not met");
    
else
    
    f = @(x) min(edf_isfd_diff(max_pts:next_zero)) + x*abs(min(edf_isfd_diff(max_pts:next_zero))/100);
    
    h = histogram(edf_isfd_diff(max_pts:next_zero), f(0:99), 'Normalization', 'cdf');
    hist_vals = h.Values;
    fiftyperc = feval(f, findLevel(hist_vals, 0.5, NaN));
    
    if isnan(fiftyperc)
        fprintf('Could not find 50 perc mark\n');
    end

    complete_idx = max_pts + findLevel(edf_isfd_diff(max_pts:next_zero), fiftyperc*complete_level, NaN);
    
    if isnan(complete_idx)
        fprintf('Error: Completeness limit not found!\n');
    else
        completeness = interp1(edf_diam, complete_idx);
    end
end

fprintf('\n\nEstimated completeness limit:    %.2f', completeness);

%% VII. Plotting

fprintf('\n\nPreparing plots');

% Prep for plotting: get relevant axis limits & indices for uncertainty fill
idx_min = find(edf_diam >= min(diam), 1, 'First');
idx_com = find(edf_diam >= completeness, 1, 'First');
idx_max = find(edf_diam <= max(diam), 1, 'Last');
yc_min  = .5 * min(edf_erro_csfd_1s_neg(idx_com:idx_max));
yc_max  = 2 * max(edf_erro_csfd_1s_pos(idx_min:idx_max));
yr_min  = .5 * min(edf_erro_rsfd_1s_neg(idx_com:idx_max));
yr_max  = 2 * max(edf_erro_rsfd_1s_pos(idx_min:idx_max));
yi_min  = .5 * min(edf_erro_isfd_1s_neg(idx_com:idx_max));
yi_max  = 2 * max(edf_erro_isfd_1s_pos(idx_min:idx_max));
yd_min  = .5 * min(edf_erro_dsfd_1s_neg(idx_com:idx_max));
yd_max  = 2 * max(edf_erro_dsfd_1s_pos(idx_min:idx_max));
c_pos   = find(edf_erro_csfd_pos);
c_neg   = find(edf_erro_csfd_neg);
r_pos   = find(edf_erro_rsfd_pos);
r_neg   = find(edf_erro_rsfd_neg);
i_pos   = find(edf_erro_isfd_pos);
i_neg   = find(edf_erro_isfd_neg);
d_pos   = find(edf_erro_dsfd_pos);
d_neg   = find(edf_erro_dsfd_neg);

if yc_min <= 0
    yc_min = 0.2;
end

if yr_min <= 0
    yr_min = 0.2;
end

if yi_min <= 0
    yi_min = 0.2;
end

if yd_min <= 0
    yd_min = 0.2;
end

% Plot 1: CSFD & RSFD
% Change the figure settings so our plot doesn't get clipped
set(0,'defaultfigurecolor',[1 1 1], 'defaultfigureposition', [0, 0, 700, 1100]);
figure( 'Name', 'CSFD & RSFD Plots' );

% Make subplots
p_csfd_rsfd(1) = subplot(3,1,1); % CSFD plot
p_csfd_rsfd(2) = subplot(3,1,2); % RSFD plot
p_csfd_rsfd(3) = subplot(3,1,3); % Rug plot

% Set axis properties
linkaxes(p_csfd_rsfd, 'x');
set(p_csfd_rsfd(1), 'XTickLabels', [], 'XScale', 'log', 'Units', 'Pixels', ...          % CSFD
    'Position', [100 399 500 600],  'YLim', [yc_min yc_max], 'YScale', ...
    'log', 'Box', 'On', 'XGrid', 'On', 'YGrid', 'On', 'FontSize', 12);
    hold(p_csfd_rsfd(1), 'on');

set(p_csfd_rsfd(2), 'XTickLabels', [], 'XScale', 'log', 'YAxisLocation',...             % RSFD
    'right', 'Units', 'Pixels', 'Position', [100 140 500 260], 'YLim',...
    [yr_min yr_max], 'YScale', 'log', 'Box', 'On', 'XGrid', 'On', ...
    'YGrid', 'On', 'FontSize', 12); hold(p_csfd_rsfd(2), 'on');

set(p_csfd_rsfd(3), 'XLim', [min(diam) max(diam)], 'XScale', 'log', ...                 % Rug
    'Units', 'Pixels', 'Position', [100 100 500 25],'YColor', 'w', ...
    'YLim', [0 10], 'FontSize', 12); hold(p_csfd_rsfd(3), 'on');

% CSFD Plot
c(1) = patch(p_csfd_rsfd(1), [edf_diam(c_neg); flipud(edf_diam(c_pos))], ...            % uncertainty fill
    [edf_erro_csfd_neg(c_neg); flipud(edf_erro_csfd_pos(c_pos))], 'r', ...
    'EdgeColor', 'None', 'FaceAlpha', 0.1);
c(2) = loglog( p_csfd_rsfd(1), edf_diam, edf_erro_csfd_1s_pos, 'r--','LineWidth', .5 ); % + 1s error
c(3) = loglog( p_csfd_rsfd(1), edf_diam, edf_erro_csfd_1s_neg, 'r--','LineWidth', .5 ); % - 1s error
c(4) = loglog( p_csfd_rsfd(1), edf_diam, edf_csfd,'r','MarkerSize',8,'LineWidth',2);    % data
c(5) = line( p_csfd_rsfd(1), [completeness completeness], p_csfd_rsfd(1).YLim, ...      % completeness
    'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');

ylabel( p_csfd_rsfd(1), ['Cumulative Frequency per ' measureUnit, '^2'], ...            % y label
    'Units', 'Pixels', 'Position', [-50 300 -1], 'FontSize', 14);
legend( p_csfd_rsfd(1), [c(4), c(2), c(1), c(5)], {'Empirical Distribution', ...        % legend, just use CSFD data since labels are same for RSFD
    '1-Sigma Confidence', 'Uncertainty', ['Completeness Limit: ' ...
    num2str(round(completeness,2)) ' ' measureUnit]}, 'FontSize', 12, 'Location', 'NorthEast');

% RSFD Plot
r(1) = patch(p_csfd_rsfd(2), [edf_diam(r_neg); flipud(edf_diam(r_pos))], ...            % uncertainty fill
    [edf_erro_rsfd_neg(r_neg); flipud(edf_erro_rsfd_pos(r_pos))], 'r', ...
    'EdgeColor', 'None', 'FaceAlpha', 0.1);
r(2) = loglog( p_csfd_rsfd(2), edf_diam, edf_erro_rsfd_1s_pos, 'r--','LineWidth', .5 ); % + 1s error
r(3) = loglog( p_csfd_rsfd(2), edf_diam, edf_erro_rsfd_1s_neg, 'r--','LineWidth', .5 ); % - 1s error
r(4) = loglog( p_csfd_rsfd(2), edf_diam, edf_rsfd,'r','MarkerSize',8,'LineWidth',2);    % data
r(5) = line( p_csfd_rsfd(2), [completeness completeness], p_csfd_rsfd(2).YLim, ...      % completeness
    'Color', 'b', 'LineWidth', 1, 'LineStyle', '--'); 

ylabel( p_csfd_rsfd(2), 'R-Plot Value', 'Color', 'k', 'Rotation', 270, ...              % y label
    'Units', 'Pixels', 'Position', [550 140 -1], 'FontSize', 14);

% Rug Plot
for i = 1:length(diam)
    loglog(p_csfd_rsfd(3), [diam(i) diam(i)], [4 7], 'r');                              % data
end
xlabel(['Diameter in ' measureUnit], 'FontSize', 14);                                   % x label
title( p_csfd_rsfd(1), 'Cumulative & Relative Size Frequencies', 'FontSize', 16);

% Plot 2: ISFD
% Change the figure settings so our plot doesn't get clipped
set(0,'defaultfigurecolor',[1 1 1], 'defaultfigureposition', [0, 0, 700, 600]);
figure( 'Name', 'ISFD Plot' );

% Make subplots
p_isfd(1) = subplot(2,1,1); % ISFD plot
p_isfd(2) = subplot(2,1,2); % Rug plot

% Set axis properties
linkaxes(p_isfd, 'x');
set(p_isfd(1), 'XTickLabels', [], 'XScale', 'log', 'Units', 'Pixels', ...               % ISFD
    'Position', [100 140 500 360],  'YLim', [yi_min yi_max], 'YScale', ...
    'log', 'Box', 'On', 'XGrid', 'On', 'YGrid', 'On', 'FontSize', 12);
    hold(p_isfd(1), 'on');

set(p_isfd(2), 'XLim', [min(diam) max(diam)], 'XScale', 'log', ...                      % Rug
    'Units', 'Pixels', 'Position', [100 100 500 25],'YColor', 'w', ...
    'YLim', [0 10], 'FontSize', 12); hold(p_isfd(2), 'on');

% ISFD Plot
inc(1) = patch(p_isfd(1), [edf_diam(i_neg); flipud(edf_diam(i_pos))], ...               % uncertainty fill
    [edf_erro_isfd_neg(i_neg); flipud(edf_erro_isfd_pos(i_pos))], 'r', ...
    'EdgeColor', 'None', 'FaceAlpha', 0.1);
inc(2) = loglog( p_isfd(1), edf_diam, edf_erro_isfd_1s_pos, 'r--','LineWidth', .5 );    % + 1s error
inc(3) = loglog( p_isfd(1), edf_diam, edf_erro_isfd_1s_neg, 'r--','LineWidth', .5 );    % - 1s error
inc(4) = loglog( p_isfd(1), edf_diam, edf_isfd,'r','MarkerSize',8,'LineWidth',2);       % data
inc(5) = line( p_isfd(1), [completeness completeness], p_isfd(1).YLim, ...              % completeness
    'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');

ylabel( p_isfd(1), ['Incremental Frequency per ' measureUnit, '^2'], ...                % y label
    'Units', 'Pixels', 'Position', [-50 175 -1], 'FontSize', 14);
legend( p_isfd(1), [inc(4), inc(2), inc(1), inc(5)], {'Empirical Distribution', ...     % legend, just use CSFD data since labels are same for RSFD
    '1-Sigma Confidence', 'Uncertainty', ['Completeness Limit: ' ...
    num2str(round(completeness,2)) ' ' measureUnit]}, 'FontSize', 12, 'Location', 'NorthEast');

% Rug Plot
for i = 1:length(diam)
    loglog(p_isfd(2), [diam(i) diam(i)], [4 7], 'r');                                   % data
end
xlabel(['Diameter in ' measureUnit], 'FontSize', 14);                                   % x label
title( p_isfd(1), 'Incremental Size Frequencies', 'FontSize', 16);

% Plot 3: DSFD
% Change the figure settings so our plot doesn't get clipped
set(0,'defaultfigurecolor',[1 1 1], 'defaultfigureposition', [0, 0, 700, 600]);
figure( 'Name', 'DSFD Plot' );

% Make subplots
p_dsfd(1) = subplot(2,1,1); % ISFD plot
p_dsfd(2) = subplot(2,1,2); % Rug plot

% Set axis properties
linkaxes(p_dsfd, 'x');
set(p_dsfd(1), 'XTickLabels', [], 'XScale', 'log', 'Units', 'Pixels', ...               % ISFD
    'Position', [100 140 500 360],  'YLim', [yd_min yd_max], 'YScale', ...
    'log', 'Box', 'On', 'XGrid', 'On', 'YGrid', 'On', 'FontSize', 12);
    hold(p_dsfd(1), 'on');

set(p_dsfd(2), 'XLim', [min(diam) max(diam)], 'XScale', 'log', ...                      % Rug
    'Units', 'Pixels', 'Position', [100 100 500 25],'YColor', 'w', ...
    'YLim', [0 10], 'FontSize', 12); hold(p_dsfd(2), 'on');

% ISFD Plot
d(1) = patch(p_dsfd(1), [edf_diam(d_neg); flipud(edf_diam(d_pos))], ...                 % uncertainty fill
    [edf_erro_dsfd_neg(d_neg); flipud(edf_erro_dsfd_pos(d_pos))], 'r', ...
    'EdgeColor', 'None', 'FaceAlpha', 0.1);
d(2) = loglog( p_dsfd(1), edf_diam, edf_erro_dsfd_1s_pos, 'r--','LineWidth', .5 );      % + 1s error
d(3) = loglog( p_dsfd(1), edf_diam, edf_erro_dsfd_1s_neg, 'r--','LineWidth', .5 );      % - 1s error
d(4) = loglog( p_dsfd(1), edf_diam, edf_dsfd,'r','MarkerSize',8,'LineWidth',2);         % data
d(5) = line( p_dsfd(1), [completeness completeness], p_dsfd(1).YLim, ...                % completeness
    'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');

ylabel( p_dsfd(1), ['Differential Frequency per ' measureUnit, '^2'], ...               % y label
    'Units', 'Pixels', 'Position', [-50 175 -1], 'FontSize', 14);
legend( p_dsfd(1), [d(4), d(2), d(1), d(5)], {'Empirical Distribution', ...             % legend, just use CSFD data since labels are same for RSFD
    '1-Sigma Confidence', 'Uncertainty', ['Completeness Limit: ' ...
    num2str(round(completeness,2)) ' ' measureUnit]}, 'FontSize', 12, 'Location', 'NorthEast');

% Rug Plot
for i = 1:length(diam)
    loglog(p_dsfd(2), [diam(i) diam(i)], [4 7], 'r');                                   % data
end
xlabel(['Diameter in ' measureUnit], 'FontSize', 14);                                   % x label
title( p_dsfd(1), 'Differential Size Frequencies', 'FontSize', 16);


%%
fprintf('\n\nFinished. ');
toc % end timing function

%% VIII. Functions

% Copycat version of findLevel in Igor
% Returns the average index of the straddling levels
% Option for failure similar to 'value = V_flag == 0 ? V_LevelX : onFail'

% Non-monotonic case code was adapted from Steffen's answer to the Stack
% Exchange Question on Matlab: non-monotonic interpolation
% https://stackoverflow.com/questions/30456014/matlab-non-monotonic-interpolation
function idx = findLevel(y, level, onFail) 
    
    % monotonic case: don't need to try as hard
    if all(diff(y) > 0) || all(diff(y) < 0) 
        
        x0 = find( y > level, 1, 'first');
        x1 = find( y <= level, 1, 'last');
    
        if x1 == x0
            idx = x0;
        else
            idx = x0 + ((level - y(x0)).*(x1-x0) ./ (y(x1)-y(x0)));
        end
        
        if isempty(idx)
            idx = onFail;
        end
    
    % not monotonic case: need to iterate through
    else
        result = [];
        x      = 1:length(y);

        for i = 1:length(y)-1
            if (y(i) <= level && level < y(i+1)) || (y(i) > level && level >= y(i+1))
                result = interp1([y(i),y(i+1)],[x(i),x(i+1)],level);
                break
            end
        end

        if isempty(result)
            idx = onFail;
        else
            idx = mean(result);    
        end
    end
end

% KDE Template function with switch case for different kernels
% except for uniform, idk why I haven't written it yet
function dsfd = kde_fun(old_dsfd, expected, measured, sd, kern, N)
	switch kern
        case 'u'    % uniform
            % ??

        case 't'    % triangular
            dsfd = old_dsfd + ( 1 / ( N*sd ) * ( 5 - abs( (expected - measured) / sd ) ) );

        case 'c'    % cosine
            dsfd = old_dsfd + ( 1 / ( N*sd ) * ( pi() / 4 * cos( pi() / 10 * (expected - measured) / sd ) ) );

        case 'e'    % epanechnikov
            dsfd = old_dsfd + ( 1 / ( N*sd ) * 3/4 * ( 1 - ( (expected - measured) / ( 5 * sd ) ) ^ 2 ));

        otherwise   % gaussian (default)
            dsfd = old_dsfd + ( 1 / (2*N*sd)) * exp( -0.5 * ((expected - measured) / sd)^2);   
                
	end
end

