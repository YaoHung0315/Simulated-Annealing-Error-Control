clc; clear; close all;

%% ---------------- Settings ----------------
N        = 100;          % 網格數
Tsteps   = 200;          % 總步數 (SA 主循環步數)
sigma    = 0.15;         % 噪音強度
delta    = 1e-6;         % clip 邊界
idx_show = 50;           % 在 feature 圖追蹤的索引

% SA 參數
p      = 0.90;           % 初始接受機率
alpha  = 0.98;           % 冷卻率 (幾何降溫: T <- alpha*T)
Kinit  = 10;             % 溫度初始化：10 次獨立一步試算

rng(42);

% h, f, g
h    = @(x) sin(x);
% f    = @(x) h(x) + sigma*randn(size(x)); % random 的做法
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
f    = @(x) h(x) + sigma*mlp_like(x); % 模擬MLP的做法

clip = @(y) min(max(y,-1+delta), 1-delta);
g    = @(y) asin( clip(y) );

% 構造一個「目標/乾淨」波作為參考（只用於畫圖）
s       = linspace(0,1,N);
x_true  = 0.5*sin(2*pi*s);
y_clean = h(x_true);

% 初始 feature / spatial
x_init = x_true + 0.10*randn(1,N);
y_init = f(x_init);

%% ---------------- 全歷程儲存 ----------------
% 1-based: 第 1 列是 t=0（初始），第 (t+1) 列是時間步 t 的結果
x_all = nan(Tsteps+1, N);
y_all = nan(Tsteps+1, N);
x_all(1,:) = x_init;
y_all(1,:) = y_init;

%% ---------------- 溫度初始化（10 次獨立一步） ----------------
% E_samples = zeros(Kinit, N);
% for k = 1:Kinit
%     y_k        = f(x_init);                % 從同一 x_cur 產生獨立 noisy y
%     E_samples(k,:) = abs(y_k - y_init);    % 只用相鄰差定義能量
% end
% E_old = mean(E_samples, 1);               % 逐點平均能量
% T     = - E_old ./ log(p);                % 逐點初始溫度
% T(~isfinite(T)) = 0;                      % 安全處理 (避免 0/0)

E_samples = zeros(Kinit, N);
x_tmp = x_init;
y_tmp = y_init;
for k = 1:Kinit
    x_tmp = g(y_tmp);              % 下一步會用到的 x
    y_next_tmp = f(x_tmp);         % 對應候選 y
    E_samples(k,:) = abs(y_next_tmp - y_tmp);  % 真一步差
    y_tmp = y_next_tmp;            % 推進暫態
end
E_old = mean(E_samples, 1);               % 逐點平均能量
T     = - E_old ./ log(p);                % 逐點初始溫度
T(~isfinite(T)) = 0;                      % 安全處理 (避免 0/0)

%% ---------------- Iteration (With SA) ----------------
x_cur = x_init;
y_cur = y_init;
for t = 1:Tsteps
    % 候選 y：一定先算出
    y_cand = f(x_cur);
    E_new  = abs(y_cand - y_cur);

    % 逐點 Metropolis 接受規則
    dE = E_new - E_old;
    acceptMask = (dE <= 0);                                   % 下降：必收
    uphill = ~acceptMask;                                     % 上升：機率接受
    if any(uphill)
        P = exp( -dE(uphill) ./ max(T(uphill), 1e-12) );
        acceptMask(uphill) = rand(1, sum(uphill)) < P;
    end

    % 形成下一狀態（僅在接受處更新）
    y_next = y_cur;
    y_next(acceptMask) = y_cand(acceptMask);
    x_next = g(y_next);

    % 只在接受處更新 E_old
    E_old(acceptMask) = E_new(acceptMask);

    % 降溫（每步幾何冷卻）
    T(acceptMask) = alpha * T(acceptMask);

    % 儲存當步完整狀態（所有點）
    x_all(t+1,:) = x_next;
    y_all(t+1,:) = y_next;

    % 前進
    x_cur = x_next;
    y_cur = y_next;
end

%% ---------------- Figure 1: Spatial ----------------
figure('Name','Spatial comparison (With SA)','Color','w');
plot(1:N, y_clean, '-k', 'LineWidth', 2); hold on;
plot(1:N, y_all(1,:), '--', 'LineWidth', 1.8);   % 初始 t=0
plot(1:N, y_all(end,:), '-', 'LineWidth', 1.8);  % 最終 t=Tsteps
grid on; box on;
xlabel('Spatial index i'); ylabel('y');
title('Spatial: Ground truth vs Iter#1 vs Final (With SA)');
legend({'Ground truth (clean y)', 'Iteration 1 (clean f(x))', ...
        sprintf('Iteration %d (clean f(x))', Tsteps)}, 'Location','best');

%% ---------------- Figure 2: Feature trajectory ----------------
xgrid = linspace(-1.3, 1.3, 400);
ygrid = h(xgrid);
cmap  = flipud(gray(Tsteps+1));

figure('Name','Feature-space trajectory (With SA)','Color','w');
plot(xgrid, ygrid, 'k-', 'LineWidth', 1.5); hold on;
grid on; box on;
xlabel('x (feature)'); ylabel('y = sin(x)');
title(sprintf('Feature space: trajectory of index %d (With SA)', idx_show));

% 強調起點與終點
scatter(x_all(1, idx_show),   h(x_all(1, idx_show)),   80, 'ro', 'filled', 'MarkerEdgeColor','k');
scatter(x_all(end, idx_show), h(x_all(end, idx_show)), 80, 'bo', 'filled', 'MarkerEdgeColor','y');

% 中間歷程（淺->深）
for t = 1:(Tsteps+1)
    xi = x_all(t, idx_show);
    yi = h(xi);
    scatter(xi, yi, 36, cmap(t,:), 'filled', 'MarkerEdgeColor','k');
end

scatter(x_all(1, idx_show),   h(x_all(1, idx_show)),   80, 'ro', 'filled', 'MarkerEdgeColor','k');
scatter(x_all(end, idx_show), h(x_all(end, idx_show)), 80, 'bo', 'filled', 'MarkerEdgeColor','y');

legend({'y = sin(x)', 'Iterations (light \rightarrow dark)', 'Start', 'End'}, 'Location','best');

%% ---------------- Figure 3: RMSE between consecutive iterations (y) -----
% y_all 的維度是 (Tsteps+1) x N，行對應時間（第1列是 t=0，最後一列是 t=Tsteps）
rmse_y = zeros(Tsteps,1);
for t = 2:(Tsteps+1)
    diff_t     = y_all(t,:) - y_all(t-1,:);
    rmse_y(t-1)= sqrt(mean(diff_t.^2));
end

figure('Name','RMSE of y across iterations','Color','w');
plot(1:Tsteps, rmse_y, '-', 'LineWidth', 1.5);
grid on; box on;
xlabel('Iteration t');
ylabel('RMSE(y^{(t)} - y^{(t-1)})');
title('Convergence diagnostics: RMSE between consecutive y');





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