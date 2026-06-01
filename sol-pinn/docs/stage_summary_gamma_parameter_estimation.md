# sol-pinn 阶段性总结与 gamma 参数识别流程

本文档总结 `sol-pinn` 项目当前已经完成的工作，并重点说明 sheath 参数 `gamma` 的参数识别流程。这里不把 `gamma` 的求解夸大为完整未知机制恢复，而是表述为：在固定一维 SOL 输运模型和其他参数的前提下，利用温度观测估计 sheath 边界参数 `gamma`。内容只描述当前代码与已有报告中已经实现或测试覆盖的部分，不扩展为尚未完成的能力。

## 1. 当前项目已经做了什么

`sol-pinn` 当前围绕一维稳态 SOL 平行热输运问题，已经完成以下模块：

1. 建立了一维 SOL 平行热输运物理模型。
2. 实现了上游温度边界与靶板 sheath 热流边界。
3. 实现了有限差分参考求解器，用于生成或对比参考温度剖面。
4. 实现了 PINN 正问题求解框架，用神经网络近似温度场 `T(s)`。
5. 实现了参数化 PINN 的基本结构，可将 `T_up` 作为输入之一。
6. 实现了 sheath 参数 `gamma` 的参数识别流程，包括：
   - 基于 PINN 温度场的后处理诊断；
   - 靶板附近窗口化诊断；
   - 最小二乘参数估计基线；
   - LS 初始化后再进行 PINN 联合参数估计的 hybrid 流程。
7. 编写了自动化测试，覆盖物理层、PINN 网络、损失函数、有限差分参考、`gamma` 诊断与参数识别流程。

当前测试结果：

```text
python -m pytest tests -q
35 passed
```

## 2. 正问题物理模型

项目研究的是沿磁力线方向的一维稳态平行热输运。空间坐标为：

```text
s in [0, L]
```

其中：

- `s = 0` 表示上游位置；
- `s = L` 表示靶板位置；
- `T(s)` 表示电子温度。

控制方程为：

```text
d/ds( kappa_parallel * T^(5/2) * dT/ds ) + S(s) = 0
```

其中：

- `T(s)`：电子温度；
- `kappa_parallel`：平行热导系数前因子；
- `T^(5/2)`：Spitzer-Harm 型热导的温度依赖；
- `S(s)`：体热源项。

在当前主要测试与参数识别示例中，常用情形是：

```text
S(s) = 0
```

此时平行热流沿程守恒。

## 3. 边界条件

### 3.1 上游温度边界

上游采用 Dirichlet 温度边界：

```text
T(0) = T_up
```

### 3.2 靶板 sheath 热流边界

靶板端采用 sheath 热流边界：

```text
-kappa_parallel * T(L)^(5/2) * dT/ds(L) = alpha * sqrt(T(L))
```

左边是传导热流：

```text
q_t = -kappa_parallel * T(L)^(5/2) * dT/ds(L)
```

右边是 sheath 允许通过的热流：

```text
q_sheath = alpha * sqrt(T(L))
```

当前代码中：

```text
alpha = gamma * e^(3/2) * p0 / (2 * sqrt(m_i))
```

令：

```text
C = e^(3/2) * p0 / (2 * sqrt(m_i))
```

则：

```text
alpha = gamma * C
```

因此靶板 sheath 边界也可以写成：

```text
-kappa_parallel * T(L)^(5/2) * dT/ds(L)
= gamma * C * sqrt(T(L))
```

## 4. 有限差分参考解

项目实现了有限差分求解器 `FDSolver`，用于求解同一个一维边值问题。

有限差分离散的主要形式为：

```text
1/h^2 * [ f_{i+1/2} * (T_{i+1} - T_i)
        - f_{i-1/2} * (T_i - T_{i-1}) ] = -S_i
```

其中：

```text
f(T) = kappa_parallel * T^(5/2)
```

界面热导使用调和平均：

```text
f_{i+1/2} = 2 * f(T_i) * f(T_{i+1}) / (f(T_i) + f(T_{i+1}))
```

上游边界：

```text
T_0 = T_up
```

靶板边界离散为：

```text
f(T_N) * (T_{N-1} - T_N) / h = alpha * sqrt(T_N)
```

对于 `S(s) = 0` 的情形，代码使用解析二点模型生成初值，并在该情形下可直接返回解析参考剖面。

## 5. PINN 正问题流程

PINN 使用神经网络近似温度场：

```text
T_theta(s) ≈ T(s)
```

当前实现包含多种网络形式：

- 普通 MLP PINN；
- Fourier feature PINN；
- 变换变量 PINN；
- 靶板区域分支增强的 Piecewise PINN；
- 输入包含 `s` 和 `T_up` 的参数化 PINN。

### 5.1 网络输入与输出

典型 PINN 输入为：

```text
s
```

网络内部通常使用归一化坐标：

```text
s_norm = s / L
```

Fourier feature 编码形式为：

```text
gamma(s_norm) = [sin(2*pi*B*s_norm), cos(2*pi*B*s_norm)]
```

网络输出通过 `softplus` 保证温度为正：

```text
T_theta(s) = T_up * softplus(NN(s/L))
```

### 5.2 PDE 残差

PINN 通过自动微分计算：

```text
dT_theta/ds
```

传导热流为：

```text
q_theta(s) = -kappa_parallel * T_theta(s)^(5/2) * dT_theta/ds
```

再通过自动微分计算：

```text
dq_theta/ds
```

由于原方程为：

```text
d/ds(kappa_parallel * T^(5/2) * dT/ds) + S(s) = 0
```

而：

```text
q = -kappa_parallel * T^(5/2) * dT/ds
```

所以代码中的残差写作：

```text
R_pde(s) = dq_theta/ds - S(s)
```

训练中使用归一化残差：

```text
R_pde_norm = R_pde / (kappa_parallel * T_up^(7/2) / L^2)
```

PDE 损失为：

```text
L_pde = mean(R_pde_norm^2)
```

### 5.3 边界损失

上游边界损失：

```text
L_up = ((T_theta(0) - T_up) / T_up)^2
```

sheath 边界损失：

```text
L_sheath =
[
  (kappa_parallel * T_theta(L)^(5/2) * dT_theta/ds(L)
   + alpha * sqrt(T_theta(L)))
  / (alpha * sqrt(T_up))
]^2
```

### 5.4 正问题总损失

正问题训练时，总损失为：

```text
L_total =
  w_pde * L_pde
+ w_up * L_up
+ w_sheath * L_sheath
```

优化流程为：

1. 使用 Adam 进行初始训练；
2. 使用 L-BFGS 进一步收敛；
3. 在评估点上输出 `T_theta(s)`；
4. 与有限差分参考解比较误差。

## 6. gamma 参数识别目标

`gamma` 参数识别的目标是：

```text
已知少量温度观测点 T_obs(s_i)，在其他模型参数给定的前提下估计 sheath 参数 gamma
```

观测数据形式为：

```text
{(s_i, T_obs_i)} for i = 1, ..., N_obs
```

在当前项目测试中，观测数据主要由有限差分参考解插值得到，也可以叠加人工噪声用于鲁棒性测试。

## 7. PINN 后处理诊断 gamma 的流程

这是当前 `gamma` 参数识别中的基本流程。

### 7.1 输入

输入包括：

```text
s_obs      观测位置
T_obs      观测温度
T_up       上游温度
L          连接长度
kappa_parallel
p0
gamma_init 初始 gamma，仅用于构造初始配置
```

### 7.2 训练温度场

先训练一个 PINN 温度场：

```text
T_theta(s)
```

参数识别训练时，默认不强制使用 sheath 边界损失：

```text
sheath_weight = 0
```

这样做的原因是：如果初始 `gamma_init` 偏离真实值，过早加入 sheath 边界可能把温度场训练到错误约束附近。

参数识别训练损失为：

```text
L_parameter_estimation =
  w_pde * L_pde
+ w_up * L_up
+ w_data * L_data
```

数据损失为：

```text
L_data = mean((T_theta(s_i) - T_obs_i)^2)
```

或使用 Huber 损失：

```text
L_data =
  mean(0.5 * diff^2)                         if |diff| <= delta
  mean(delta * (|diff| - 0.5 * delta))        if |diff| > delta
```

其中：

```text
diff = T_theta(s_i) - T_obs_i
```

### 7.3 计算靶板端温度与梯度

训练完成后，在靶板端计算：

```text
T_L = T_theta(L)
```

并通过自动微分计算：

```text
dT_ds_L = dT_theta/ds at s = L
```

### 7.4 由 sheath 关系反推 gamma

靶板热流为：

```text
q_t = -kappa_parallel * T_L^(5/2) * dT_ds_L
```

sheath 关系为：

```text
q_t = gamma * C * sqrt(T_L)
```

其中：

```text
C = e^(3/2) * p0 / (2 * sqrt(m_i))
```

因此：

```text
gamma = q_t / (C * sqrt(T_L))
```

代入 `q_t`：

```text
gamma =
  -kappa_parallel * T_L^(5/2) * dT_ds_L
  / (C * sqrt(T_L))
```

化简为：

```text
gamma =
  -kappa_parallel * T_L^2 * dT_ds_L
  / C
```

这对应代码中的 `diagnose_gamma_from_sheath()`。

## 8. 窗口化 gamma 诊断流程

单点使用 `dT/ds(L)` 对导数误差较敏感。项目因此实现了靶板附近窗口化诊断。

### 8.1 取靶板附近窗口

选取：

```text
s in [L * (1 - window_ratio), L]
```

例如：

```text
window_ratio = 0.1
```

表示使用最后 10% 的空间区间。

### 8.2 在窗口内计算热流

对窗口内多个点 `s_j`，计算：

```text
T_j = T_theta(s_j)
```

```text
dT_ds_j = dT_theta/ds at s = s_j
```

```text
q_j = -kappa_parallel * T_j^(5/2) * dT_ds_j
```

### 8.3 用窗口热流估计 gamma

当前 `S(s) = 0` 测试问题中，热流理论上沿程守恒，因此可用窗口内多个热流估计值诊断 `gamma`：

```text
gamma_j = q_j / (C * sqrt(T_L))
```

然后取：

```text
gamma = median(gamma_j)
```

或：

```text
gamma = mean(gamma_j)
```

当前默认更偏向使用 median，以减少局部异常导数对结果的影响。

这对应代码中的 `diagnose_gamma_from_window()`。

## 9. 最小二乘参数估计基线

项目还实现了最小二乘参数估计作为基线方法。

### 9.1 思路

给定一个候选 `gamma`，用有限差分正问题求解器得到：

```text
T_fd(s; gamma)
```

然后在观测点比较：

```text
T_fd(s_i; gamma) - T_obs_i
```

通过优化 `gamma` 最小化温度残差平方和：

```text
min_gamma sum_i [T_fd(s_i; gamma) - T_obs_i]^2
```

### 9.2 用途

当前项目中，最小二乘方法有两个用途：

1. 作为与 PINN 参数识别结果对比的传统基线；
2. 在 noisy case 中为 PINN 联合参数估计提供 `gamma` 初值或先验。

## 10. Hybrid 参数估计流程

对于有噪声数据，项目实现了 LS 初始化加 PINN 联合优化的 hybrid 方案。

### 10.1 第一步：LS 初步估计

先用最小二乘参数估计得到：

```text
gamma_prior
```

### 10.2 第二步：设置可训练 gamma

代码中使用：

```text
log_gamma
```

作为可训练参数，并令：

```text
gamma = exp(log_gamma)
```

这样可以保证：

```text
gamma > 0
```

### 10.3 第三步：联合训练温度场和 gamma

联合参数估计损失为：

```text
L_joint =
  L_pde
+ L_up
+ L_data
+ w_sheath * L_trainable_sheath
+ w_prior * (log_gamma - log(gamma_prior))^2
```

其中可训练 sheath 损失在靶板附近窗口上计算：

```text
L_trainable_sheath =
mean[
  (
    kappa_parallel * T_theta(s_j)^(5/2) * dT_theta/ds(s_j)
    + gamma * C * sqrt(T_theta(s_j))
  )^2
] / scale^2
```

这里：

```text
C = e^(3/2) * p0 / (2 * sqrt(m_i))
```

`scale` 用于量纲归一化，代码中使用与 sheath 热流量级相关的尺度。

### 10.4 输出

训练结束后输出：

```text
gamma = exp(log_gamma)
```

对应代码中的：

```text
train_with_hybrid_inversion()
```

## 11. 当前已有结果的保守表述

根据当前文档与测试，项目可以保守表述为：

1. 在一维简化 SOL sheath 模型上，已经实现了 PINN 正问题求解与有限差分参考解对比流程。
2. 在合成观测数据上，已经实现了从温度观测估计 `gamma` 的流程。
3. 在干净合成数据中，当前参数识别测试可以得到接近设定值的 `gamma`。
4. 在加入噪声后，单点 PINN 后处理诊断会变敏感。
5. 窗口化诊断、Huber 数据损失、LS 初始化和 hybrid 联合参数估计已经作为改进策略加入代码。
6. 当前测试覆盖了诊断公式、窗口诊断、可训练 sheath 损失、最小二乘初始化以及若干参数识别场景。

## 12. 当前没有完成或不应夸大的部分

当前项目尚不能表述为已经完成真实装置上的 SOL/divertor 预测工具。原因包括：

1. 当前主模型仍是一维稳态简化模型。
2. 当前参数识别主要基于合成观测数据。
3. 当前物理闭合较简化，例如使用等压相关假设和简化 sheath 热流关系。
4. 当前还没有接入真实实验诊断数据。
5. 当前还没有替代 SOLPS、DIV1D 等高保真边界等离子体模拟工具。
6. 噪声数据下的 PINN 参数识别仍需要进一步增强鲁棒性。

因此，当前阶段更准确的定位是：

```text
一个用于验证 PINN 在一维 SOL 平行热输运和 sheath 参数识别中可行性的研究型原型。
```
