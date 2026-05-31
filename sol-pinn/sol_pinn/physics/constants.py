"""物理常数和默认参数值.

所有单位采用 SI + eV 混合体系:
- 温度: eV
- 长度: m
- 热导率: W/(m·eV^(7/2))  (Spitzer-Härm 平行热导系数)
"""

# Spitzer-Härm 平行热导系数 [W/(m·eV^(7/2))]
# 对于电子, κ_∥ ≈ 2000 / (Z_eff * lnΛ) 量级
# 此处使用典型值 (对应 Z_eff=1, lnΛ≈15)
KAPPA_0 = 2000.0

# Sheath 热传输系数 (无量纲)
# 典型范围 γ ≈ 5-8, 默认取 7.0
# 定义: q_t = γ · e · n_t · T_t · sqrt(T_t / m_i)
GAMMA_SHEATH = 7.0

# 基本物理常数
E_CHARGE = 1.602e-19       # 元电荷 [C]
M_I = 2.0 * 1.673e-27      # 氘离子质量 [kg] (D⁺)

# 偏滤器参考参数 (conduction-limited regime)
L_DEFAULT = 20.0            # 沿磁力线长度 [m] (更长SOL路径→传导限制区)
T_UP_DEFAULT = 80.0         # 上游电子温度 [eV]
P0_DEFAULT = 2.0e21         # 静压常数 [eV/m³] (更高密度→增强sheath热损失)

# 数值保护
T_EPSILON = 1e-8            # 温度下限, 防除零和数值不稳定
