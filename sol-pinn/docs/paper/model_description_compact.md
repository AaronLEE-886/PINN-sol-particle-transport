# 刮削层（SOL）平行输运问题的物理建模与 PINN 求解

## 物理背景

托卡马克聚变装置中，芯部等离子体沿磁力线流向偏滤器靶板，形成**刮削层（Scrape-Off Layer, SOL）**。SOL 中的平行能量输运决定了靶板热负载分布，是偏滤器设计的关键问题。

简化为一维模型：沿磁力线方向 $s \in [0, L]$，$s=0$ 为上游（芯部边界），$s=L$ 为偏滤器靶板。输运流程为：芯部等离子体 → 沿磁力线平行输运 → 刮削层 SOL → 偏滤器靶板 → 鞘层边界 → 粒子回收与热负载。

## 控制方程

稳态 SOL 平行输运由非线性热传导方程的描述：

$$
\frac{d}{ds}\left(\kappa_\parallel T^{5/2} \frac{dT}{ds}\right) + S(s) = 0, \quad s \in [0, L]
$$

- $T(s)$ — 电子温度 [eV]
- $\kappa_\parallel$ — Spitzer-Härm 平行热导系数 [W/(m·eV$^{7/2}$)]
- $S(s)$ — 体积热源项 [W/m³]

Spitzer-Härm 热导率呈强温度非线性：$\kappa_\parallel(T) = \kappa_\parallel \cdot T^{5/2}$。

## 边界条件

**上游边界**（$s=0$）为固定温度 Dirichlet 条件：

$$
T(0) = T_{\text{up}}
$$

**靶板边界**（$s=L$）由磁鞘物理决定。离子进入鞘层需达到声速 $c_s = \sqrt{e(T_e+T_i)/m_i}$（$T_e=T_i=T$ 时 $c_s = \sqrt{2eT/m_i}$）。靶板粒子流和能流：

$$
\Gamma = n_t c_s,\quad q_{\text{sheath}} = \gamma \Gamma T_t = \gamma n_t T_t c_s
$$

$\gamma \approx 7$ 为鞘层热传输系数。

### 鞘层热流系数 $\alpha$

将 $q_{\text{sheath}}$ 从 [eV·m⁻²s⁻¹] 转换为 [W/m²] 并代入 $c_s$：

$$
q_{\text{sheath}} = e \gamma n_t T_t \sqrt{\frac{2eT_t}{m_i}} = \gamma n_t \sqrt{\frac{2e^3}{m_i}} T_t^{3/2}
$$

引入等压假设 $p_0 = n_t T_t$ 消去 $n_t$：

$$
q_{\text{sheath}} = \gamma p_0 \sqrt{\frac{2e^3}{m_i}} T_t^{1/2} \equiv \alpha \sqrt{T_t},\qquad
\boxed{\alpha = \gamma \sqrt{\frac{2e^3}{m_i}} p_0}
$$

代码中声速采用 $c_s = \sqrt{eT_e/m_i}$，因此 $\alpha_{\text{code}} = \gamma e^{3/2} p_0 / (2\sqrt{m_i})$，相差因子 $\sqrt{2}$。

### 边界条件完整形式

靶板处平行热导输运的热流与鞘层允许通过的热流平衡：

$$
\boxed{-\kappa_\parallel T(L)^{5/2} \left.\frac{dT}{ds}\right|_{s=L} = \alpha \sqrt{T(L)}}
$$

## 解析解（$S=0$）

$S=0$ 时热流守恒 $\kappa_\parallel T^{5/2} dT/ds = -q$（常数）。分离变量积分：

$$
\kappa_\parallel \int_{T_{\text{up}}}^{T(s)} T^{5/2} dT = -q \int_0^s ds' \;\Longrightarrow\; \frac{2}{7}\kappa_\parallel \left[T(s)^{7/2} - T_{\text{up}}^{7/2}\right] = -q s
$$

解得温度分布：

$$
T(s) = \left(T_{\text{up}}^{7/2} - \frac{7}{2}\frac{q}{\kappa_\parallel} s\right)^{2/7},\qquad q = \alpha \sqrt{T(L)}
$$

代入 $s=L$ 得靶板温度自洽方程：

$$
\boxed{T_t^{7/2} + \frac{7\alpha L}{2\kappa_\parallel} T_t^{1/2} = T_{\text{up}}^{7/2}},\qquad C = \frac{7\alpha L}{2\kappa_\parallel},\quad T_t^{3.5} + C T_t^{0.5} = T_{\text{up}}^{3.5}
$$

可用二分法高效求解，是 FD 求解器的初值生成基础。

## 有限差分（FD）求解

对方程进行有限差分离散：

$$
\frac{1}{h^2}\left[f_{i+1/2}(T_{i+1} - T_i) - f_{i-1/2}(T_i - T_{i-1})\right] = -S_i
$$

界面处 $\kappa_\parallel T^{5/2}$ 采用调和平均：

$$
f_{i+1/2} = \frac{2 f(T_i) f(T_{i+1})}{f(T_i) + f(T_{i+1})}, \quad f(T) = \kappa_\parallel T^{5/2}
$$

边界离散：

- 上游：$T_0 = T_{\text{up}}$
- 靶板：$f(T_N)(T_{N-1} - T_N) / h = \alpha \sqrt{T_N}$

## PINN 求解

PINN 用神经网络 $T_\theta(s)$ 近似温度分布，通过将 PDE 残差和边界条件作为损失函数训练，使预测自动满足物理约束。

### 网络结构

前向传播路径：$s \to \tilde{s} = s/L \to$ Fourier 编码 $\to$ MLP $\to$ 输出变换。

**Fourier 特征编码**：直接将 $s$ 输入 MLP 会产生低频偏好（谱偏差），而 SOL 靶板附近梯度极陡需要高频分量。编码将输入映射到高频空间：

$$
\gamma(\tilde{s}) = \left[\sin(2\pi \mathbf{B} \tilde{s}),\; \cos(2\pi \mathbf{B} \tilde{s})\right]^T \in \mathbb{R}^{2m}
$$

其中 $\mathbf{B} \in \mathbb{R}^{m \times 1}$ 固定，$B_{ij} \sim \mathcal{N}(0, \sigma^2)$。超参数 $\sigma$ 控制频率分布宽度：$\sigma=5.0$ 适合传导限制区（陡梯度），$\sigma=1.0$ 适合鞘层限制区（平滑解）。$m=64$ 输出 128 维特征。

**MLP 隐藏层**：编码后的特征送入全连接网络 $\mathbf{h}_{k+1} = \tanh(\mathbf{W}_k \mathbf{h}_k + \mathbf{b}_k)$。激活函数用 $\tanh$ 因为其二阶可导——ReLU 二阶导数为零无法表示 PDE 残差。最优结构：$D=3$ 层，每层 64 神经元（两个 regime 均如此），默认配置为 $5 \times 128$。

**输出变换**：最后一层经线性变换后通过 softplus 保证正性，再乘以 $T_{\text{up}}$：

$$
T_\theta(s) = T_{\text{up}} \cdot \text{softplus}\left(\mathbf{W}_{\text{out}} \mathbf{h}_{D-1} + b_{\text{out}}\right)
$$

输出偏置 $b_{\text{out}}$ 初始化为 $\ln(e^{1/2} - 1)$，使初始预测约 $T_{\text{up}} / 2$，有利于训练初期稳定。

### 损失函数

总损失为三项加权和，默认 $w_{\text{pde}} = w_{\text{up}} = w_{\text{sheath}} = 1.0$：

$$
\mathcal{L} = w_{\text{pde}} \cdot \mathcal{L}_{\text{PDE}} + w_{\text{up}} \cdot \mathcal{L}_{\text{up}} + w_{\text{sheath}} \cdot \mathcal{L}_{\text{sheath}}
$$

**PDE 残差损失**（自动微分，特征尺度归一化）：

$$
\mathcal{L}_{\text{PDE}} = \frac{1}{N_c}\sum_{i=1}^{N_c} \left[\frac{1}{\kappa_\parallel T_{\text{up}}^{3.5}/L^2} \cdot \frac{d}{ds}\left(\kappa_\parallel T_\theta^{5/2}\frac{dT_\theta}{ds}\right)\bigg|_{s_i}\right]^2
$$

归一化因子 $\kappa_\parallel T_{\text{up}}^{3.5} / L^2$ 使残差量级为 $\mathcal{O}(1)$。

**上游边界损失**：$\mathcal{L}_{\text{up}} = \big((T_\theta(0) - T_{\text{up}})/T_{\text{up}}\big)^2$

**鞘层边界损失**（自动微分求 $dT/ds$）：

$$
\mathcal{L}_{\text{sheath}} = \left[\frac{\kappa_\parallel T_\theta(L)^{5/2} \cdot dT_\theta/ds(L) + \alpha\sqrt{T_\theta(L)}}{\alpha \sqrt{T_{\text{up}}}}\right]^2
$$

分子为零时恰好满足 sheath 边界条件，分母 $\alpha \sqrt{T_{\text{up}}}$ 为热流特征尺度用于归一化。

### 训练策略

两阶段优化。**Phase 1 — Adam**（lr=1e-3，3000–15000 步）：快速降低损失，逃离差局部极小。**Phase 2 — L-BFGS**（强 Wolfe 线搜索，200–500 步）：利用曲率信息精细收敛。

![Training Flow](../figures/pinn/training_flow.png)

### 训练流程

**Step 1 — 生成配点**：调用 `target_refined(200, 100, L, boundary_ratio=0.10)` 生成 200 个全域均匀点 + 100 个靶板附近 $[0.9L, L]$ 加密点，确保梯度最陡区域获得更多约束。

**Step 2 — 准备张量**：转为 PyTorch 浮点张量 $(N_c, 1)$，使后续自动微分可求 $dT_\theta/ds$ 和 $d^2T_\theta/ds^2$。

**Step 3 — 模型状态**：调用 `model.train()` 启用训练模式，清除优化器过往状态。

**Step 4 — Adam 迭代**（3000–15000 步），每步执行：

4a. **前向传播**：$s_i \to$ 网络 $\to T_\theta(s_i)$，自动追踪计算图
4b. **计算损失**：自动微分求 $dT_\theta/ds$ 和 $d^2T_\theta/ds^2$，构造 PDE 残差 $\to \mathcal{L}_{\text{PDE}}$；取 $T_\theta(0)$ 计算 $\mathcal{L}_{\text{up}}$；取 $T_\theta(L)$ 和 $dT_\theta/ds(L)$ 构造 sheath 残差 $\to \mathcal{L}_{\text{sheath}}$；三项加权求和得 $\mathcal{L}$
4c. **反向传播**：`loss.backward()` 计算 $\partial\mathcal{L}/\partial\theta$
4d. **优化器更新**：`optimizer.step()` 更新权重 → `zero_grad()` 清梯度
4e. **监控**：每 1000 步打印损失值

**Step 5 — 切换 L-BFGS**：构造 L-BFGS 优化器，传入闭包函数（返回当前损失和梯度）。

**Step 6 — L-BFGS 迭代**（200–500 步）：闭包内执行与 Step 4a–4c 相同的前向和反向传播。L-BFGS 利用历史梯度差近似 Hessian 曲率，配合强 Wolfe 线搜索确定搜索方向和步长，200–500 步内收敛到比 Adam 更优的局部极小。

**Step 7 — 评估模式**：`model.eval()` + `torch.no_grad()`，后续预测仅一次前向传播，不计梯度、不占内存。

### 配点策略

均匀采样适合鞘层限制区（$T(s)$ 平坦）；靶板加密采样（$s \in [0.9L, L]$ 区域密集配点）对传导限制区至关重要。推荐 `target_refined(200, 100, L, boundary_ratio=0.10)`。

## 关键参数

| 参数       |         符号         |              典型值              |  单位  |
| ---------- | :------------------: | :------------------------------: | :----: |
| 上游温度   |  $T_{\text{up}}$  |             40–200             |   eV   |
| 磁力线长度 |        $L$        |              10–20              |   m   |
| 热导系数   | $\kappa_\parallel$ |            1000–2000            |   —   |
| 静压       |       $p_0$       |  $1\text{–}2\times 10^{21}$  | eV/m³ |
| 鞘层系数   |      $\gamma$      |               7.0               |   —   |
| 鞘层流系数 |      $\alpha$      | $3.88\text{–}7.76\times 10^6$ |   —   |

## T_up 扫描结果

| $T_{\text{up}}$ (eV) | $T_t$ (eV) | $T_t/T_{\text{up}}$ |   $q_t$ (W/m²)   | 非线性程度 |
| :--------------------: | :----------: | :-------------------: | :-----------------: | :--------: |
|           40           |     0.56     |         0.014         | $5.78\times 10^6$ |    极强    |
|           50           |     2.65     |         0.053         | $1.26\times 10^7$ |     强     |
|           60           |     9.46     |         0.158         | $2.39\times 10^7$ |    明显    |
|           70           |    26.16    |         0.374         | $3.97\times 10^7$ |  中等偏强  |
|           80           |    48.52    |         0.607         | $5.39\times 10^7$ |    中等    |
|           90           |    67.05    |         0.745         | $6.35\times 10^7$ |  中等偏弱  |
|          100          |    82.36    |         0.824         | $7.04\times 10^7$ |     弱     |
|          150          |    142.85    |         0.952         | $9.27\times 10^7$ |  接近线性  |
|          200          |    196.06    |         0.980         | $1.09\times 10^8$ |  几乎线性  |

## PINN 精度

| $T_{\text{up}}$ |      相对 L2 误差      | 最大绝对误差 | 状态 |
| :---------------: | :--------------------: | :----------: | :--: |
|        40        | $3.51\times 10^{-1}$ |   14.86 eV   |  ❌  |
|        50        | $1.47\times 10^{-2}$ |   3.31 eV   | ⚠️ |
|        60        | $2.95\times 10^{-3}$ |   0.36 eV   |  ✅  |
|        70        | $5.08\times 10^{-5}$ |   0.007 eV   |  ✅  |
|        80        | $1.09\times 10^{-5}$ |   0.002 eV   |  ✅  |
|        90+        | $< 1\times 10^{-5}$ |  < 0.002 eV  |  ✅  |

低 $T_{\text{up}}$ 时误差大的原因：

1. **动态范围跨 2 个数量级**（40 eV → 0.56 eV）
2. **靶板梯度极陡**：$dT/ds \approx -109$ eV/m
3. **Fourier 高频分量不足**或**配点未在边界加密**

根本原因在于 Spitzer-Härm 的正反馈：$T \downarrow \to \kappa_{\text{eff}} \downarrow \to dT/ds \uparrow$，温度越低梯度越陡，网络越难分辨。

**改进方向**：靶板加密配点、增大 Fourier $\sigma$（2.0–5.0）、自适应配点、残差注意力重加权。

## 符号对照表

|         符号         | 含义           |                  典型值                  |        单位        |
| :------------------: | :------------- | :---------------------------------------: | :-----------------: |
|        $T$        | 电子温度       |                 0.56–200                 |         eV         |
|        $s$        | 沿磁力线坐标   |                   0–20                   |          m          |
|        $L$        | SOL 连接长度   |                  10–20                  |          m          |
| $\kappa_\parallel$ | 平行热导系数   |                1000–2000                | W/(m·eV$^{7/2}$) |
|      $\alpha$      | 鞘层热流系数   | $3.88 \times 10^6$–$7.76\times 10^6$ |         —         |
|      $\gamma$      | 鞘层热传输系数 |                    7.0                    |         —         |
|       $p_0$       | 静压常数       | $1 \times 10^{21}$–$2\times 10^{21}$ |       eV/m³       |
|        $q$        | 平行热流密度   | $5.78\times 10^6$–$1.09\times 10^8$ |        W/m²        |
|       $m_i$       | 氘离子质量     |     $2 \times 1.673\times 10^{-27}$     |         kg         |
|        $e$        | 元电荷         |         $1.602\times 10^{-19}$         |          C          |
