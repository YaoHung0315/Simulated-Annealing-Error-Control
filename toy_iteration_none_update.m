clc; clear; close all;

%% ---------------- Settings ----------------
N        = 100;                     % 網格數
Tsteps   = 500;                     % 總步數
sigma    = 0.15;                    % 噪音強度 (固定)
delta    = 1e-6;                    % clip 邊界
idx_show = 50;                      % 在 feature 圖追蹤的索引

lambdaArr = [0, 0.3, 0.6, 0.9, 1.0]; % 掃描的 under-relaxation
lam_plot  = 0.6;                    % 用於 Figure 2 的 lambda

rng(42);

% h, f, g
h    = @(x) sin(x);
f    = @(x) h(x) + sigma*randn(size(x)); % random 的做法
% f    = @(x) h(x) + sigma*( ...
%       0.35*sin(3.1*x) ...
%     - 0.27*cos(2.3*x) ...
%     + 0.19*tanh(1.7*x) ...
%     + 0.23*erf(1.1*x) ...
%     + 0.15*sin(x.^2) ...
%     - 0.21*cos(x.^3) ...
%     + 0.12*x + 0.08*x.^2 - 0.05*x.^3 ...
%     + 0.18*sin( (0.7*x + 0.5*sin(2.4*x)).^2 ) ...
%     + 0.14*tanh( (x + 0.3*cos(3.7*x)).*(0.6 + 0.2*sin(1.9*x)) ) ...
%     + 0.11*exp(-0.8*x.^2).*sin(4.3*x + 0.5*cos(1.3*x)) ...
%     - 0.09*erf(0.7*x + 0.2*sin(2.2*x)).*cos(3.6*x) ...
% ); % 自訂function的做法
% f    = @(x) h(x) + sigma*mlp_like(x); % 模擬MLP的做法

clip = @(y) min(max(y,-1+delta), 1-delta);
g    = @(y) asin( clip(y) );

% Ground truth 波形（導引與畫圖）
s       = linspace(0,1,N);
x_true  = 0.5*sin(2*pi*s);
y_clean = h(x_true);

% 共同初始條件
x_init = x_true + 0.10*randn(1,N);
y_init = f(x_init);

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

%% ---------------- 主迴圈（只模擬 + 存檔） ----------------
for lam = lambdaArr
    x_cur = x_init;  
    y_cur = y_init;
    for t = 1:Tsteps
        y_cand = f(x_cur);
        y_next = y_cand;
        x_next = (1 - lam)*x_true + lam*g(y_next);

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

        x_cur = x_next;  y_cur = y_next;
    end
end

%% ---------------- 顏色 ----------------
colors = lines(numel(lambdaArr));

%% ---------------- Figure 1: Spatial（所有 λ 的最終曲線） ----------------
figure('Name','Spatial comparison (No SA, GT-guided)','Color','w'); hold on; grid on; box on;
plot(1:N, y_clean,  '-k', 'LineWidth', 2);
plot(1:N, y_all_00(1,:), '--', 'Color',[0.4 0.4 0.4], 'LineWidth',1.5); % 初始
% 逐 λ 畫最終曲線
plot(1:N, y_all_00(end,:), '-', 'Color', colors(1,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.0');
plot(1:N, y_all_03(end,:), '-', 'Color', colors(2,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.3');
plot(1:N, y_all_06(end,:), '-', 'Color', colors(3,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.6');
plot(1:N, y_all_09(end,:), '-', 'Color', colors(4,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.9');
plot(1:N, y_all_10(end,:), '-', 'Color', colors(5,:), 'LineWidth',1.6, 'DisplayName','\lambda=1.0');
xlabel('Spatial index i'); ylabel('y');
title('Spatial: Ground truth vs Initial vs Final (No SA, all \lambda)');
legend({'Ground truth (clean y)','Initial (t=0)','\lambda=0.0','\lambda=0.3','\lambda=0.6','\lambda=0.9','\lambda=1.0'}, 'Location','best');

%% ---------------- Figure 2: Feature trajectory（只畫 lam_plot） ---------
% 取對應 x_all_* 的歷程
switch lam_plot
    case 0.0, x_hist = x_all_00(:,idx_show);
    case 0.3, x_hist = x_all_03(:,idx_show);
    case 0.6, x_hist = x_all_06(:,idx_show);
    case 0.9, x_hist = x_all_09(:,idx_show);
    case 1.0, x_hist = x_all_10(:,idx_show);
end
xgrid = linspace(-1.3, 1.3, 400); ygrid = h(xgrid); cmap = flipud(gray(Tsteps+1));

figure('Name','Feature-space trajectory (No SA, GT-guided)','Color','w'); hold on; grid on; box on;
plot(xgrid, ygrid, 'k-', 'LineWidth', 1.5);
xlabel('x (feature)'); ylabel('y = sin(x)');
title(sprintf('Feature space: trajectory of index %d (No SA, \\lambda = %.1f)', idx_show, lam_plot));
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
% 以相鄰時間步的 y 計算 RMSE
rmse_00 = sqrt(mean((y_all_00(2:end,:) - y_all_00(1:end-1,:)).^2, 2));
rmse_03 = sqrt(mean((y_all_03(2:end,:) - y_all_03(1:end-1,:)).^2, 2));
rmse_06 = sqrt(mean((y_all_06(2:end,:) - y_all_06(1:end-1,:)).^2, 2));
rmse_09 = sqrt(mean((y_all_09(2:end,:) - y_all_09(1:end-1,:)).^2, 2));
rmse_10 = sqrt(mean((y_all_10(2:end,:) - y_all_10(1:end-1,:)).^2, 2));

figure('Name','RMSE of y across iterations (No SA, GT-guided)','Color','w'); hold on; grid on; box on;
semilogy(1:Tsteps, rmse_00, '-', 'Color', colors(1,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.0');
semilogy(1:Tsteps, rmse_03, '-', 'Color', colors(2,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.3');
semilogy(1:Tsteps, rmse_06, '-', 'Color', colors(3,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.6');
semilogy(1:Tsteps, rmse_09, '-', 'Color', colors(4,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.9');
semilogy(1:Tsteps, rmse_10, '-', 'Color', colors(5,:), 'LineWidth',1.6, 'DisplayName','\lambda=1.0');
xlabel('Iteration t'); ylabel('RMSE(y^{(t)} - y^{(t-1)})');
title('Convergence diagnostics: RMSE between consecutive y (No SA, all \lambda)');
legend show;

%% ---------------- Figure 4: Convergence vs \lambda（mean |x-x_true|） ----
err_00 = mean(abs(x_all_00 - x_true), 2);
err_03 = mean(abs(x_all_03 - x_true), 2);
err_06 = mean(abs(x_all_06 - x_true), 2);
err_09 = mean(abs(x_all_09 - x_true), 2);
err_10 = mean(abs(x_all_10 - x_true), 2);

figure('Name','Convergence vs lambda (No SA, GT-guided)','Color','w'); hold on; grid on; box on;
semilogy(0:Tsteps, err_00, '-', 'Color', colors(1,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.0');
semilogy(0:Tsteps, err_03, '-', 'Color', colors(2,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.3');
semilogy(0:Tsteps, err_06, '-', 'Color', colors(3,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.6');
semilogy(0:Tsteps, err_09, '-', 'Color', colors(4,:), 'LineWidth',1.6, 'DisplayName','\lambda=0.9');
semilogy(0:Tsteps, err_10, '-', 'Color', colors(5,:), 'LineWidth',1.6, 'DisplayName','\lambda=1.0');
xlabel('Iteration'); ylabel('mean |x^{(t)} - x_{true}|  (log scale)');
title('No SA: ground-truth-guided under-relaxation convergence (all \lambda)');
legend show;






function y = mlp_like(x)
% 一個固定權重的兩層 MLP：phi(x) → tanh → tanh → 線性輸出
% 完全確定、無隨機；權重用解析公式產生（與索引相關），避免手打巨矩陣。

x = x(:)';                          % 保證為 1×N
% 特徵展開（高「維度感」的確定性非線性特徵）
Phi = [ ...
    x; x.^2; x.^3; ...
    sin(1.7*x); sin(3.1*x); sin(x.^2); ...
    cos(2.3*x); cos(4.9*x); cos(x.^3); ...
    tanh(1.4*x); tanh(0.7*x); ...
    erf(0.9*x); ...
    exp(-0.6*x.^2); exp(-0.2*x.^2).*sin(5.3*x) ...
];                                  % d×N, 其中 d=14

d  = size(Phi,1);
H1 = 24; H2 = 16;                   % 隱藏層寬度

% 用解析式生成固定權重（與索引相關，確定性且較「雜」）
[i1,j1] = ndgrid(1:H1,1:d);
W1 = 0.25*sin( sqrt(2)*i1 + sqrt(3)*j1 ) + 0.15*cos( 0.7*i1.*j1 );
b1 = 0.10*sin( (1:H1)*sqrt(5) );

[i2,j2] = ndgrid(1:H2,1:H1);
W2 = 0.20*cos( sqrt(7)*i2 + 0.9*j2 ) + 0.12*sin( 0.5*i2.*j2 );
b2 = 0.08*cos( (1:H2)*sqrt(11) );

w3 = 0.18*sin( (1:H2)*sqrt(13) ) + 0.12*cos( (1:H2)*sqrt(17) );
b3 = 0.05;

% 前向傳播（向量化）
Z1 = tanh( W1*Phi + b1.' );
Z2 = tanh( W2*Z1  + b2.' );
y  = (w3*Z2 + b3);                  % 1×N
end