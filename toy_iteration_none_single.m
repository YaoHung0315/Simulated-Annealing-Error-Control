%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% toy_noSA.m
% 單點 ML–CFD 耦合 (無 SA) ：觀察 under-relaxation λ 對收斂影響
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ------------------ 基本設定 -------------------------------------------
maxIter   = 100;          % 最大迭代步
sigma     = 0.05;         % ML 震盪強度 ε ~ N(0,σ²)
lambdaArr = [0 0.3 0.6 1.0];% 掃描多個 under-relaxation 係數
xFinal    = pi/6;         % 目標固定點
x0        = -pi/4;        % 初始特徵

rng(42);                  % 隨機種子

%% ------------------ 圖形初始化 ----------------------------------------
figure('Name','No SA','Position',[60 60 660 480]);
hold on; grid on; box on;
colors = lines(numel(lambdaArr));

%% ------------------ 逐 λ 模擬 -----------------------------------------
for kLam = 1:numel(lambdaArr)
    lam = lambdaArr(kLam);
    
    x   = x0;
    e   = nan(maxIter,1);
    e(1)= abs(x - xFinal);
    
    for t = 2:maxIter
        eps_t = sigma * randn;           % ML 震盪
        y_t   = sin(x) + eps_t;          % f(x)
        
        % CFD 回推 (無 SA → 必接受)
        x     = (1-lam)*xFinal + lam*asin(y_t);
        e(t)  = abs(x - xFinal);
    end
    
    semilogy(e,'-','Color',colors(kLam,:), ...
             'LineWidth',1.4, ...
             'DisplayName',sprintf('\\lambda = %.1f',lam));
end

title(sprintf('No SA  (\\sigma = %.3f)', sigma));
xlabel('Iteration'); ylabel('|x^{(t)} - x_{final}|  (log scale)');
legend show;