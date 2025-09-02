clc; clear; close all;

%% ---------------- Settings ----------------
N        = 100;                     % 網格數
Tsteps   = 1000;                     % 總步數
sigma    = 0.25;                    % 噪音強度 (固定)
delta    = 1e-6;                    % clip 邊界
idx_show = 50;                      % 在 feature 圖追蹤的索引

lambdaArr = [0, 0.3, 0.6, 0.9, 1.0]; % 掃描的 under-relaxation
lam_plot  = 0.6;                    % 用於 Figure 2 的 lambda

% SA 參數
p      = 0.90;                     % 初始接受機率
alpha  = 0.98;                     % 冷卻率 (幾何降溫)
Kinit  = 10;                       % 溫度初始化：10 次獨立一步

rng(42);

% h, f, g
h    = @(x) sin(x);

% --- 時間序列 s_vec：若不存在或長度不足就生成 ---
Ttotal    = Tsteps + Kinit + 10;             % 確保足夠長
min_needed = Kinit + Tsteps + 1;             % ★ 不依賴 t0，避免未定義
if ~exist('s_vec','var') || numel(s_vec) < min_needed
    s_vec  = s_multi_sine(Ttotal, 6);        % 或改用 s_logistic / s_rudin_shapiro
end

phi = @(x) mlp_like(x);
x_probe = linspace(-1.3, 1.3, 2048);
mu_phi  = mean(phi(x_probe));
phi0    = @(x) (phi(x) - mu_phi);

% 可選軟夾，避免 asin 飽和（如不需要可刪）
% k_soft = pi/2; softclip = @(y) (1 - delta) * (2/pi) * atan(k_soft * y);
% f = @(x,t) softclip( h(x) + sigma * phi0(x) * s_vec(t) );

f = @(x,t) h(x) + sigma * phi0(x) * s_vec(t);

clip = @(y) min(max(y,-1+delta), 1-delta);
g    = @(y) asin( clip(y) );

% Ground truth 波形（導引與畫圖）
s       = linspace(0,1,N);
x_true  = 0.5*sin(2*pi*s);
y_clean = h(x_true);

% 共同初始條件
x_init = x_true + 0.10*randn(1,N);

% === 初始 ===
t0    = 1;                       % ★ 改成 1（MATLAB 1-based 索引）
y_init = f(x_init, t0);

%% ---------------- 五個二維陣列（lambda 專屬） ----------------
x_all_00 = nan(Tsteps+1, N);  y_all_00 = nan(Tsteps+1, N);
x_all_03 = nan(Tsteps+1, N);  y_all_03 = nan(Tsteps+1, N);
x_all_06 = nan(Tsteps+1, N);  y_all_06 = nan(Tsteps+1, N);
x_all_09 = nan(Tsteps+1, N);  y_all_09 = nan(Tsteps+1, N);
x_all_10 = nan(Tsteps+1, N);  y_all_10 = nan(Tsteps+1, N);

% t=0 初始同一組
x_all_00(1,:)=x_init; y_all_00(1,:)=y_init;
x_all_03(1,:)=x_init; y_all_03(1,:)=y_init;
x_all_06(1,:)=x_init; y_all_06(1,:)=y_init;
x_all_09(1,:)=x_init; y_all_09(1,:)=y_init;
x_all_10(1,:)=x_init; y_all_10(1,:)=y_init;

%% ---------------- 主迴圈（SA：只模擬 + 存檔） ----------------
for lam = lambdaArr
    x_cur = x_init;  
    y_cur = y_init;

    % ---- 溫度初始化（10 次獨立一步，不推進狀態）----
    E_samples = zeros(Kinit, N);
    x_tmp = x_init;
    y_tmp = y_init;
    for k = 1:Kinit
        x_tmp = g(y_tmp);              % 下一步會用到的 x
        y_next_tmp = f(x_tmp, t0 + k);
        E_samples(k,:) = abs(y_next_tmp - y_tmp);  % 真一步差
        y_tmp = y_next_tmp;            % 推進暫態
    end
    epsE  = 1e-10;
    E_old = 100*ones(1,100);%max(mean(E_samples, 1), epsE);          % 能量下限
    T     = max(- E_old ./ log(p), 1e-8);           % 溫度下限

    for t = 1:Tsteps
        % 候選（先算出，不一定採用）
        y_cand = (1 - lam)*y_clean + lam*f( g(y_cur), t0 + Kinit + t );  % ★ 接在 Kinit 之後
        E_new  = abs(y_cand - y_cur);

        % 逐點 Metropolis 接受規則
        dE = E_new - E_old;
        acceptMask = (dE <= 0);
        upMask = ~acceptMask;
        if any(upMask)
            P = exp( -dE(upMask) ./ max(T(upMask), 1e-12) );
            acceptMask(upMask) = rand(1,sum(upMask)) < P;
        end

        % 更新（僅在接受處）
        y_next = y_cur;
        y_next(acceptMask) = y_cand(acceptMask);
        x_next = g(y_next);

        E_old(acceptMask) = E_new(acceptMask);
        T(acceptMask) = alpha * T(acceptMask);   % 幾何降溫

        % 存入對應的二維陣列
        switch lam
            case 0.0
                x_all_00(t+1,:)=x_next; y_all_00(t+1,:)=y_next;
            case 0.3
                x_all_03(t+1,:)=x_next; y_all_03(t+1,:)=y_next;
            case 0.6
                x_all_06(t+1,:)=x_next; y_all_06(t+1,:)=y_next;
            case 0.9
                x_all_09(t+1,:)=x_next; y_all_09(t+1,:)=y_next;
            case 1.0
                x_all_10(t+1,:)=x_next; y_all_10(t+1,:)=y_next;
        end

        % 前進
        x_cur = x_next;  y_cur = y_next;
    end
end

%% ---------------- 顏色 ----------------
colors = lines(numel(lambdaArr));

%% ---------------- Figure 1: Spatial（所有 λ 的最終曲線） ----------------
figure('Name','Spatial comparison (With SA, GT-guided)','Color','w'); hold on; grid on; box on;
plot(1:N, y_clean,  '-k', 'LineWidth', 2);
plot(1:N, y_all_00(1,:), '--', 'Color',[0.4 0.4 0.4], 'LineWidth',1.5); % 初始
% 逐 λ 畫最終曲線
plot(1:N, y_all_00(end,:), '-', 'Color', colors(1,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.0');
plot(1:N, y_all_03(end,:), '-', 'Color', colors(2,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.3');
plot(1:N, y_all_06(end,:), '-', 'Color', colors(3,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.6');
plot(1:N, y_all_09(end,:), '-', 'Color', colors(4,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.9');
plot(1:N, y_all_10(end,:), '-', 'Color', colors(5,:), 'LineWidth',1.6, 'DisplayName','\lambda=1.0');
xlabel('Spatial index i'); ylabel('y');
title('Spatial: Ground truth vs Initial vs Final (With SA, all \lambda)');
legend({'Ground truth (clean y)','Initial (t=0)','\lambda=0.0','\lambda=0.3','\lambda=0.6','\lambda=0.9','\lambda=1.0'}, 'Location','best');

%% ---------------- Figure 2: Feature trajectory（只畫 lam_plot） ---------
switch lam_plot
    case 0.0, x_hist = x_all_00(:,idx_show);
    case 0.3, x_hist = x_all_03(:,idx_show);
    case 0.6, x_hist = x_all_06(:,idx_show);
    case 0.9, x_hist = x_all_09(:,idx_show);
    case 1.0, x_hist = x_all_10(:,idx_show);
end
xgrid = linspace(-1.3, 1.3, 400); ygrid = h(xgrid); cmap = flipud(gray(Tsteps+1));

figure('Name','Feature-space trajectory (With SA, GT-guided)','Color','w'); hold on; grid on; box on;
plot(xgrid, ygrid, 'k-', 'LineWidth', 1.5);
xlabel('x (feature)'); ylabel('y = sin(x)');
title(sprintf('Feature space: trajectory of index %d (With SA, \\lambda = %.1f)', idx_show, lam_plot));
scatter(x_hist(1),   h(x_hist(1)),   80, 'ro', 'filled', 'MarkerEdgeColor','k'); % Start
scatter(x_hist(end), h(x_hist(end)), 80, 'bo', 'filled', 'MarkerEdgeColor','y'); % End
for t = 2:Tsteps
    xi = x_hist(t); yi = h(xi);
    scatter(xi, yi, 36, cmap(t,:), 'filled', 'MarkerEdgeColor','k');
end
scatter(x_hist(1),   h(x_hist(1)),   80, 'ro', 'filled', 'MarkerEdgeColor','k'); % Start
scatter(x_hist(end), h(x_hist(end)), 80, 'bo', 'filled', 'MarkerEdgeColor','y'); % End
legend({'y = sin(x)','Start','End','Iterations (light \rightarrow dark)'}, 'Location','best');

%% ---------------- Figure 3: RMSE(y)（所有 λ；顏色一致） -----------------
rmse_00 = sqrt(mean((y_all_00(2:end,:) - y_all_00(1:end-1,:)).^2, 2));
rmse_03 = sqrt(mean((y_all_03(2:end,:) - y_all_03(1:end-1,:)).^2, 2));
rmse_06 = sqrt(mean((y_all_06(2:end,:) - y_all_06(1:end-1,:)).^2, 2));
rmse_09 = sqrt(mean((y_all_09(2:end,:) - y_all_09(1:end-1,:)).^2, 2));
rmse_10 = sqrt(mean((y_all_10(2:end,:) - y_all_10(1:end-1,:)).^2, 2));

figure('Name','RMSE of y across iterations (With SA, GT-guided)','Color','w'); hold on; grid on; box on;
semilogy(1:Tsteps, rmse_00, '-', 'Color', colors(1,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.0');
semilogy(1:Tsteps, rmse_03, '-', 'Color', colors(2,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.3');
semilogy(1:Tsteps, rmse_06, '-', 'Color', colors(3,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.6');
semilogy(1:Tsteps, rmse_09, '-', 'Color', colors(4,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.9');
semilogy(1:Tsteps, rmse_10, '-', 'Color', colors(5,:), 'LineWidth',1.6, 'DisplayName','\lambda=1.0');
xlabel('Iteration t'); ylabel('RMSE(y^{(t)} - y^{(t-1)})');
title('Convergence diagnostics: RMSE between consecutive y (With SA, all \lambda)');
legend show;

%% ---------------- Figure 4: Convergence vs \lambda（mean |x-x_true|） ----
err_00 = mean(abs(x_all_00 - x_true), 2);
err_03 = mean(abs(x_all_03 - x_true), 2);
err_06 = mean(abs(x_all_06 - x_true), 2);
err_09 = mean(abs(x_all_09 - x_true), 2);
err_10 = mean(abs(x_all_10 - x_true), 2);

figure('Name','Convergence vs lambda (With SA, GT-guided)','Color','w'); hold on; grid on; box on;
semilogy(0:Tsteps, err_00, '-', 'Color', colors(1,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.0');
semilogy(0:Tsteps, err_03, '-', 'Color', colors(2,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.3');
semilogy(0:Tsteps, err_06, '-', 'Color', colors(3,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.6');
semilogy(0:Tsteps, err_09, '-', 'Color', colors(4,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.9');
semilogy(0:Tsteps, err_10, '-', 'Color', colors(5,:), 'LineWidth',1.6, 'DisplayName','\lambda=1.0');
xlabel('Iteration'); ylabel('mean |x^{(t)} - x_{true}|  (log scale)');
title('With SA: ground-truth-guided convergence (all \lambda)');
legend show;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Local functions                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = mlp_like(x)
x = x(:)';                          
Phi = [ ...
    x; x.^2; x.^3; ...
    sin(1.7*x); sin(3.1*x); sin(x.^2); ...
    cos(2.3*x); cos(4.9*x); cos(x.^3); ...
    tanh(1.4*x); tanh(0.7*x); ...
    erf(0.9*x); ...
    exp(-0.6*x.^2); exp(-0.2*x.^2).*sin(5.3*x) ...
];                                  
d  = size(Phi,1);
H1 = 24; H2 = 16;                   
[i1,j1] = ndgrid(1:H1,1:d);
W1 = 0.25*sin( sqrt(2)*i1 + sqrt(3)*j1 ) + 0.15*cos( 0.7*i1.*j1 );
b1 = 0.10*sin( (1:H1)*sqrt(5) );
[i2,j2] = ndgrid(1:H2,1:H1);
W2 = 0.20*cos( sqrt(7)*i2 + 0.9*j2 ) + 0.12*sin( 0.5*i2.*j2 );
b2 = 0.08*cos( (1:H2)*sqrt(11) );
w3 = 0.18*sin( (1:H2)*sqrt(13) ) + 0.12*cos( (1:H2)*sqrt(17) );
b3 = 0.05;
Z1 = tanh( W1*Phi + b1.' );
Z2 = tanh( W2*Z1  + b2.' );
y  = (w3*Z2 + b3);                 
end

function s = s_multi_sine(T, K)
phi = (1+sqrt(5))/2;                
alph = mod(phi.^(1:K), 1);          
t = (1:T);
S = zeros(1,T);
for k = 1:K
    S = S + sin(2*pi*alph(k)*t);
end
S = S / K;                          
s = postprocess_zero_mean_unit(S);  
end

function s = s_logistic(T)
x = 0.123456789;                     
S = zeros(1,T);
for t = 1:T
    x = 4 * x * (1 - x);             
    S(t) = x;
end
S = 2*(S - 0.5);                     
s = postprocess_zero_mean_unit(S);   
end

function s = s_rudin_shapiro(T)
S = zeros(1,T);
for n = 0:T-1
    b = dec2bin(n);
    c = 0;
    for i = 1:length(b)-1
        if b(i)=='1' && b(i+1)=='1', c = c + 1; end
    end
    S(n+1) = 1 - 2*mod(c,2);         
end
s = postprocess_zero_mean_unit(S);    
end

function s = postprocess_zero_mean_unit(S)
S = S - mean(S);
mx = max(abs(S));
if mx < 1e-12, s = S; return; end
s = S / mx;                           
end