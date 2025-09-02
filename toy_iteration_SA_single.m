%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% toy_withSA.m
% 單點 ML–CFD 耦合 (含 SA) ：能量 E = |Δx|，Metropolis + 降溫
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ------------------ 基本設定 -------------------------------------------
maxIter   = 1000;
sigma     = 0.05;
lambdaArr = [0 0.3 0.6 1.0];
xFinal    = pi/6;
x0        = -pi/4;

% SA 參數
T0    = 1.0;      % 初始溫度
alpha = 0.99;    % 冷卻率  T_{t+1}=αT_t

rng(42);

%% ------------------ 圖形初始化 ----------------------------------------
figure('Name','With SA','Position',[80 80 680 500]);
hold on; grid on; box on;
colors = lines(numel(lambdaArr));

%% ------------------ 逐 λ 模擬 -----------------------------------------
for kLam = 1:numel(lambdaArr)
    lam = lambdaArr(kLam);
    
    x    = x0;
    e    = nan(maxIter,1);
    Eold = NaN;      % 前一步能量
    T    = T0;       % 當前溫度
    
    e(1) = abs(x - xFinal);
    
    for t = 2:maxIter
        eps_t = sigma * randn;
        y_t   = sin(x) + eps_t;                  % ML 輸出
        
        xCand = (1-lam)*xFinal + lam*asin(y_t);  % CFD 回推 (候選)
        Ecand = abs(xCand - x);                  % 能量 |Δx|
        if isnan(Eold), Eold = Ecand; end        % 首步初始化
        
        % -------- Metropolis 接受規則 ----------
        if Ecand <= Eold
            accept = true;                       % 能量下降 → 必接受
        else
            P = exp(-(Ecand - Eold)/T);
            accept = rand < P;                  % 依機率接受
        end
        
        % -------- 更新狀態/拒絕處理 ------------
        if accept
            x = xCand;
            Eold = Ecand;
        end
        
        % -------- 降溫 -------------------------
        T = alpha * T;
        
        % -------- 記錄誤差 --------------------
        e(t) = abs(x - xFinal);
    end
    
    semilogy(e,'-','Color',colors(kLam,:), ...
             'LineWidth',1.4, ...
             'DisplayName',sprintf('\\lambda = %.1f',lam));
end

title(sprintf('With SA  (\\sigma = %.3f,  T_0 = %.1f, \\alpha = %.3f)', ...
               sigma,T0,alpha));
xlabel('Iteration'); ylabel('|x^{(t)} - x_{final}|  (log scale)');
legend show;