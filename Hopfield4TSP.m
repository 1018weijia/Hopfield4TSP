%% 连续Hopfield神经网络的优化—旅行商问题优化计算
clear all
clc
%% 定义全局变量
global pathConstraintWeight
% 控制路径约束，确保每个城市被访问一次且每个时间步只有一个城市被选择

global distanceConstraintWeight 
% 控制距离约束，加权距离项，影响路径的选择，使得网络优化路径长度

%% 初始化城市位置和距离
cities_num = 30; % 城市数量

cities = rand(cities_num, 2); % citys为城市的坐标矩阵

distance = dist(cities, cities'); % 计算城市间距离

%% 初始化Hopfield网络
pathConstraintWeight = 200; % 路径选择惩罚项
distanceConstraintWeight = 100; % 城市间距离惩罚项

% 超参数
step = 0.0001; % 更新步长
iter_num = 10000; % 网络迭代次数

U0 = 0.1; % 用于初始化神经元状态
% delta 为随机矩阵，其值在 [-1, 1] 之间，用于扰动 U 的初始化
delta = 2 * rand(cities_num, cities_num) - 1; 
E = zeros(1, iter_num); % 初始化能量函数的记录矩阵 E，存储每次迭代的能量值

% 初始化 U 神经元状态矩阵，基础值加上扰动项
U = U0 * log(cities_num - 1) + delta; 
% 通过 Sigmoid 函数计算初始激活状态 V
V = (1 + tansig(U / U0)) / 2;  

%% 寻优迭代
for i = 1:iter_num  
    % 动态方程计算
    dU = diff_u(V, distance);
    % 输入神经元状态更新
    U = U + dU * step;
    % 输出神经元状态更新
    V = (1 + tansig(U / U0)) / 2;
    % 能量函数计算
    E(i) = energy(V, distance);  
end

%% 判断路径有效性
final_V = zeros(cities_num, cities_num); % 最终的路径选择矩阵

[V_max, V_ind] = max(V); % 获取权重最大值索引

for j = 1:cities_num
    final_V(V_ind(j), j) = 1; % 将最大值设为选择
end

C = sum(final_V, 1); % 判断每一行，列是不是只有一个1
R = sum(final_V, 2);

% 每一行和每一列都是单独的1则表示路径有效
flag = isequal(C, ones(1, cities_num)) & isequal(R', ones(1, cities_num));

%% 结果显示
if flag == 1
   % 计算初始路径长度
   sort_rand = randperm(cities_num); % 随机选择城市
   cities_rand = cities(sort_rand, :);
   len_before = dist(cities_rand(1, :),cities_rand(end, :)');

   for i = 2:size(cities_rand,1)
       len_before = len_before + dist(cities_rand(i-1, :), cities_rand(i, :)');
   end

   % 绘制初始路径
   figure(1)
   plot([cities_rand(:,1); ...
       cities_rand(1,1)],[cities_rand(:,2); ...
       cities_rand(1,2)],'o-','LineWidth',2);

   % 绘制城市点
   for i = 1:length(cities)
       text(cities(i,1),cities(i,2),['   ' num2str(i)])
   end
   text(cities_rand(1,1),cities_rand(1,2),'      起点' )
   text(cities_rand(end,1),cities_rand(end,2),'       终点' )

   % 绘制标题
   title(['优化前路径(长度：' num2str(len_before) ')'])
   axis([0 1 0 1])
   grid on
   xlabel('城市位置横坐标')
   ylabel('城市位置纵坐标')

   % 计算最优路径长度
   [V1_max, V1_ind] = max(final_V);
   cities_final = cities(V1_ind, :);
   len_after = dist(cities_final(1, :),cities_final(end, :)');

   for i = 2:size(cities_final,1)
       len_after = len_after + dist(cities_final(i-1, :), cities_final(i, :)');
   end

   disp('最优路径矩阵');
   % 绘制最优路径
   figure(2)
   plot([cities_final(:,1); ...
       cities_final(1,1)],...
       [cities_final(:,2); ...
       cities_final(1,2)],'o-','LineWidth',2);

   % 绘制城市点
   for i = 1:length(cities)
       text(cities(i,1),cities(i,2),['  ' num2str(i)])
   end
   text(cities_final(1,1),cities_final(1,2),'       起点' )
   text(cities_final(end,1),cities_final(end,2),'       终点' )
   title(['优化后路径(长度：' num2str(len_after) ')'])
   axis([0 1 0 1])
   grid on
   xlabel('城市位置横坐标')
   ylabel('城市位置纵坐标')

   % 绘制能量函数变化曲线
   figure(3)
   plot(1:iter_num, E, 'LineWidth', 2);
   ylim([0 2000])
   title(['能量函数变化曲线(最优能量：' num2str(E(end)) ')']);
   xlabel('迭代次数');
   ylabel('能量函数');
else
   disp('寻优路径无效');
end
%% 获取网络更新参数
function du=diff_u(V,d)
% V：为网络
% d：为城市间距离
global pathConstraintWeight distanceConstraintWeight
n = size(V, 1); % 获取城市数量

% 对 V 的每一行求和，得到每个城市被访问的次数
% 减去 1 的目的是使得路径约束和选择约束为1，当不为1时，偏差越大，loss就越大
% 网络的调整程度也就越大
sum_x=repmat(sum(V,2) - 1, 1, n); 
sum_i=repmat(sum(V,1) - 1, n, 1);

% 构建符合TSP约束的循环路径，这里将第1列加到最后，使 V_temp 保持闭环结构
V_temp = V(:,2:n);
V_temp = [V_temp V(:,1)];

sum_d = d * V_temp;
du = - (pathConstraintWeight * sum_x + ... % 每个城市访问次数为1
    pathConstraintWeight * sum_i + ... % 每次只能访问一个城市
    distanceConstraintWeight * sum_d); % 要求距离越小越好
end
%% 计算能量函数
function E=energy(V,d)
% V：为网络
% d：为城市间距离
global pathConstraintWeight distanceConstraintWeight

n=size(V,1);% 获取城市数量

sum_x=sumsqr(sum(V,2) - 1); % 获得约束项的平方值
sum_i=sumsqr(sum(V,1) - 1);

V_temp=V(:,2:n);
V_temp=[V_temp V(:,1)];

sum_d= d * V_temp;
sum_d = sum(sum(V.*sum_d)); % 对距离进行求和
E=0.5*(pathConstraintWeight * sum_x + ...
    pathConstraintWeight * sum_i + ...
    distanceConstraintWeight * sum_d); % 求和得到总能量
end
