# ALGOS · 秀丽虫数字实现 · 设计文档 v0.3

> 项目代号: ALGOS-Celegans
> 文档版本: 0.3 (2026-05-20)
> 前版本: v0.2 (2026-05-20), drift_design_draft.md, celegans_design_draft.md
> v0.3 修订: Phase 0 完成后的回填——CTRNN 激活函数改为 tanh、Cook 2019
> 实测连接数补齐、SensoryTranslator 的 grounding 原则显式化、新增第 12
> 章说明与 OpenWorm 等外部资源的关系。其余结构不变。

---

## 0. 文档地位

这份文档是项目的主参考。所有实施决策应回溯到本文档对应章节。本文档的结构反映了项目的完整心智模型,实施者可以局部阅读但应该掌握整体结构。

本文档不包含:
- 完整的具体代码(在配套的 phase0_implementation.md 等阶段任务书里)
- 文献引用的详细审计(在配套的 data_audit.md 里,待建立)
- 实验结果与迭代记录(在项目日志里)

本文档严格包含:
- 项目的理论锚点和不可妥协的设计原则
- 系统架构的完整说明
- 关键模块的接口规范
- 明确的设计选择和不做什么的承诺
- 待解决的开放问题清单

---

## 1. 项目本质

### 1.1 这不是什么

- 不是秀丽虫仿真器。我们不试图精确复现秀丽虫的每一个生物细节。OpenWorm 在做那件事,他们做得比我们能做的好。
- 不是行为复制项目。复现真实秀丽虫的行为是副产物,不是目标。
- 不是工具型 AI。这个系统不为任何外部任务优化,不接受 reward signal。

### 1.2 这是什么

这是 ALGOS 框架在秀丽虫规模上的具体实例化。我们用秀丽虫的 connectome 作为有限的"演化先验",在这个先验上叠加 ALGOS 框架的核心机制(autopoiesis、grounded 感觉、有限可塑性、肉体层抽象),观察会涌现什么。

更准确地说:这是一个测试,测试一个通用的 autopoietic 数字生物框架能在 4 亿年演化馈赠的拓扑上跑出多接近真实秀丽虫的东西。差距告诉我们框架还缺什么,符合的部分告诉我们生物细节里有多少是必要的。

### 1.3 核心理论锚点

整个项目建立在四个不可妥协的理论承诺上:

承诺一: 真实的 autopoiesis

系统的持续依赖于系统自身的活动。能死,死了就死了,不可备份不可复活。每只数字虫子是一个独立个体,有自己的一生。这条不是哲学姿态,是 grounding 的根基——只有真的能死的系统才能拥有真正 grounded 的意义。

承诺二: grounded 感觉

神经系统接收的所有信号都必须 grounded 在 autopoietic 路径上。温度信号不是"温度是多少",是"温度偏离我能存活的范围多少"。化学信号不是"化学物质浓度",是"接近食物的预测信号"。grounding 通过肉体层实现,而不是神经系统内的额外机制。

承诺三: 始终在线的局部可塑性

学习不是训练阶段的产物,是系统正在活着这件事的副产物。可塑性始终在线,不能开关。但可塑性是局部规则,不是全局优化——只用 Hebbian 形式的本地更新,不用反向传播,不用任何外部 oracle。

承诺四: 多时间尺度结构

快变量(神经活动)、中变量(肌肉张力、调质浓度)、慢变量(能量稳态、可塑性效应)必须在不同的时间尺度上运行。这不是性能优化,是 autopoietic 系统的结构必要条件。同步更新的系统不能产生分层的预测。

### 1.4 评估的真实标准

主指标(必须满足):
- autopoietic 持存能力 — 系统能在环境中维持自身多久
- 扰动恢复能力 — 系统在受到非致命干扰后能恢复到稳定行为的能力

副指标(有则更好):
- 行为复现度 — 系统的行为谱与真实秀丽虫的吻合程度
- 神经活动相关性 — 内部神经元活动与真实电生理数据的相关性

如果主指标和副指标矛盾——数字虫子为了生存找到了真实虫子没有的行为策略——以主指标为准。这反映了项目立场:我们做的是 autopoietic 系统,不是秀丽虫复制品。

---

## 2. 系统架构

### 2.1 三层结构

┌─────────────────────────────────────────────┐
│ 环境 Environment                              │
│ 物理空间 / 化学场 / 温度场 / 食物源 / 危险源     │
└─────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────┐
│ 肉体 Body                                    │
│ ├─ 代谢系统(能量池、消耗、摄食、死亡判定)        │
│ ├─ 物理状态(位置、姿态、肌肉张力、身体形态)      │
│ ├─ 感觉转换(环境 + 内感受 → 神经元输入)         │
│ └─ 运动执行(神经元 → 肌肉 → 形变 → 位移)        │
└─────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────┐
│ 神经系统 Neural System                        │
│ ├─ 连接矩阵(W_chem, W_gap, M_k)              │
│ ├─ CTRNN 动力学                              │
│ ├─ 神经调质浓度                               │
│ └─ Hebbian 可塑性(掩码限制)                   │
└─────────────────────────────────────────────┘
### 2.2 三层的职责严格分离

神经系统:
- 接收 N 维输入(N = 302),输出 N 维状态
- 演化内部状态 V(t)
- 更新可塑性连接的权重
- 不知道自己有多少能量,不知道何时死亡,不知道自己有身体
- 是纯动力系统,无内部目标

肉体:
- 管理代谢、能量、死亡
- 把环境状态翻译为神经系统能理解的输入信号
- 把神经系统的输出翻译为对环境的物理作用
- grounding 在肉体到神经的接口处实现
- 是物质代谢机器,神经系统寄生其上

环境:
- 提供物理空间和资源
- 提供化学场、温度场等
- 不知道虫子,只提供"位置 → 状态"查询接口
- 是被动的状态空间

### 2.3 一个 tick 的标准流程

def tick(t):
    # 1. 肉体读取神经系统的运动输出,执行物理运动
    movement_intensity = body.motor.execute(neural.V, body.physical)
    
    # 2. 肉体检查摄食、损伤、其他物理事件
    body.check_food_intake(env)
    body.check_damage(env)
    
    # 3. 肉体更新代谢,判定死亡
    alive = body.metabolism.tick(
        neural_activity=mean(abs(neural.V)),
        movement_intensity=movement_intensity
    )
    if not alive:
        return DEAD
    
    # 4. 肉体生成新的感觉输入(grounded 在此实现)
    sensory_input = body.sensory.translate(env, body.physical, body.metabolism)
    
    # 5. 神经系统接收输入,更新状态
    neural.step(sensory_input)
    
    # 6. 神经系统更新调质浓度(可能不是每 tick)
    if t % MODULATOR_RATE == 0:
        neural.update_modulators()
    
    # 7. 神经系统应用 Hebbian 学习(始终在线但有掩码)
    neural.apply_plasticity()
    
    # 8. 环境演化(独立时间尺度)
    if t % WORLD_RATE == 0:
        env.step()
    
    return ALIVE
---

## 3. 神经系统设计

### 3.1 矩阵骨架

四个核心矩阵,N = 302:

W_chem  : ndarray(N, N)    # 化学突触矩阵
W_gap   : ndarray(N, N)    # 电突触矩阵(对称)
P       : ndarray(N, N)    # 可塑性掩码(bool 或 0/1)
S       : ndarray(N, K)    # 神经元对 K 种调质的敏感度
数据来源 (详见 data_audit.md):
- W_chem, W_gap: Cook et al. 2019 connectome,从 WormWiring 下载
- P: 文献整合,只对有证据可塑的突触位置为 1(预计 50-100 个连接)
- S: 从 WormBase 受体表达数据整合,神经元 i 对调质 k 的敏感度

符号规则:
- W_chem[i, j] 表示从 j 到 i 的化学突触
- 正值=兴奋(来自胆碱能/谷氨酸能突触前神经元)
- 负值=抑制(来自 GABA 能突触前神经元)
- W_gap[i, j] = W_gap[j, i] 总是正值

Cook 2019(corrected July 2020)实测数字 (Phase 0 加载后):

- 神经元数: 302(雌雄同体)
- 化学突触: 3,709 有向 pair(其中 ~135 抑制性 GABA 输出)
- 电突触: 1,091 unique pair(对称矩阵,nnz = 2,182)
- 类别分布: 83 sensory / 81 interneuron / 108 motor / 20 pharyngeal /
  8 sex-specific(HSN×2、VC×6)/ 2 other(CANL、CANR)
- 矩阵稀疏度: W_chem 4.07%、W_gap 2.39%

White 1986 早期估计的 ~7,000 chem / ~600 gap 数字已过时,Cook 2019
(corrected)是当前标准。两者差异主要来自 Cook 2019 对单一 EM 切片
的非重复计数与对模糊连接的更严格判定。

### 3.2 状态变量

V       : ndarray(N,)      # 神经元状态向量,值域 [-1, 1]
c       : ndarray(K,)      # K 种调质的全局浓度,K = 5
K=5 种调质: 多巴胺、血清素、章鱼胺、酪胺、神经肽 F(简化代表组)。

### 3.3 CTRNN 动力学

每个 tick 的更新方程:

def neural_step(V, c, sensory_input, W_chem, W_gap, S):
    # 1. 化学突触输入(tanh 天然中心化在 0)
    chem_input = W_chem @ np.tanh(BETA * V)
    
    # 2. 电突触输入(隐式方法保证稳定性)
    # 公式: gap_current = sum_j W_gap[i,j] * (V[j] - V[i])
    gap_input = W_gap @ V - V * W_gap.sum(axis=1)
    
    # 3. 调质对神经元兴奋性的影响
    modulator_drive = S @ c
    
    # 4. 噪声
    noise = np.random.randn(N) * NOISE_LEVEL
    
    # 5. CTRNN 更新
    dV = (-V + chem_input + gap_input + sensory_input + modulator_drive + noise) / tau
    V_new = V + dV
    V_new = np.clip(V_new, -1, 1)
    
    return V_new
参数:
- `tanh(β*V)`,β ≈ 5。tanh 的输出在 [-1, 1],V=0 处输出 0,使
  V=0 在无驱动且无重复输入的极限下是真正的不动点。数学上等价于
  `2*σ(2*β*V) - 1`(σ 为标准 logistic),因此沿用之前对 σ 的稳定性分析。
- tau: 时间常数。**初版统一为 10 tick**。后期可差异化为感觉/中间/运动三组。
- NOISE_LEVEL: 0.01

> v0.3 修订说明: Phase 0 实施时发现,直接用 `σ(V) = 1/(1+e^{-βV})` 会引入
> 一个 +0.5 的恒定偏置,经过几乎全兴奋的 W_chem 累积后令网络在 ~100 tick 内
> 饱和。Phase 0 用了 `σ(V) - 0.5` 作为临时修正;v0.3 正式把激活函数改写为
> `tanh(β*V)`,这是同一类的中心化非线性,无需 magic 常数,且与
> Beer/Izquierdo 一系 CTRNN 文献的写法一致。

### 3.4 神经调质系统

调质浓度有自己的慢动力学:

```python
def update_modulators(V, c):
    """每 MODULATOR_RATE tick 调用一次"""
    for k in range(K):
        # 浓度的稳态值 = 产生它的神经元活动
        producers = MODULATOR_PRODUCERS[k]
        target = np.mean([V[i] for i in producers])
        
        # 一阶滞后:c 缓慢趋向 target
        c[k] += (target - c[k]) / MODULATOR_TAU[k]
    
    return c
```

调质参数:
- `MODULATOR_TAU`: [100, 300, 200, 200, 500] (tick 数,衰减时间)
- `MODULATOR_PRODUCERS`: 字典,每种调质对应产生它的神经元列表

调质对神经元的影响通过敏感度矩阵 S 实现:神经元 i 接收 `(S @ c)[i]` 作为额外的兴奋输入。S[i, k] 可正可负:正值=调质 k 兴奋神经元 i,负值=抑制。

S 矩阵的具体值: 见 data_audit.md。初版可以做简化——只把已知有强证据的"调质 → 神经元"关系填入,其余位置为 0。

### 3.5 可塑性

Hebbian 形式,**只在可塑性掩码标记的连接上应用**:

```python
def apply_plasticity(V, V_prev, W_chem, P):
    # 局部 Hebbian:共激活强化连接
    # delta_W[i,j] ∝ V[i] * V_prev[j]
    correlation = np.outer(V, V_prev)
    
    # 衰减项防止权重爆炸
    decay = W_chem * DECAY_RATE
    
    # 只更新有可塑性的连接
    delta_W = (LEARNING_RATE * correlation - decay) * P
    
    W_chem += delta_W
    
    # 保持初始连接拓扑(W_chem 中原本为 0 的位置保持为 0)
    # P 已经只覆盖有连接的位置,这里是双重保险
    return W_chem
```

参数:
- `LEARNING_RATE`: 1e-4(以 tick 计的小步长,使可塑性效应在 ~10^4 tick 显现)
- `DECAY_RATE`: 1e-5

**关键性质**:
- 可塑性始终在线(每个 tick 都更新,但效应缓慢累积)
- 只更新 P 中标记的连接(预计 < 5% 的总连接)
- 大部分连接(95%+)保持 connectome 给定的初始值不变

### 3.6 时间常数和频率

所有时间用 tick:

```python
TAU_NEURAL = 10          # 神经元状态衰减时间常数
TAU_MODULATOR = [100, 300, 200, 200, 500]  # 五种调质
MODULATOR_RATE = 20      # 每 20 tick 更新调质
LEARNING_RATE = 1e-4
DECAY_RATE = 1e-5
```

### 3.7 明确不包含的

- **自我建模层**: 秀丽虫版本不包含。这是有意的设计选择,作为后续 Drift 实验的对照——测试"无自我建模"系统能跑多远。
- **任何全局误差信号或外部 reward**: 严格禁止。学习只通过 Hebbian。
- **任何形式的批量训练**: 系统始终在线,只有持续运行,没有"训练阶段"。

---

## 4. 肉体层设计

### 4.1 模块组织

```python
class Body:
    metabolism: Metabolism       # 代谢核心
    physical: PhysicalState      # 物理状态
    sensory: SensoryTranslator   # 感觉转换
    motor: MotorExecutor         # 运动执行
    intake: FoodIntake           # 摄食判定
    damage: DamageMonitor        # 损伤监测
```

### 4.2 Metabolism(代谢核心)

```python
class Metabolism:
    energy_pool: float           # 当前总能量, [0, capacity]
    energy_capacity: float       # 储备上限
    
    BASE_RATE: float = ...       # 基础代谢消耗(每 tick)
    NEURAL_COST: float = ...     # 神经系统的能量消耗(每 tick)
    MOVE_COST_FACTOR: float = ...# 运动消耗系数
    
    def tick(self, neural_activity, movement_intensity):
        # 神经系统总是消耗能量(不可关闭)
        cost = self.NEURAL_COST
        # 基础代谢
        cost += self.BASE_RATE
        # 运动消耗
        cost += self.MOVE_COST_FACTOR * movement_intensity
        
        self.energy_pool -= cost
        
        return self.energy_pool > 0  # 死亡判定
    
    def eat(self, amount):
        self.energy_pool = min(self.energy_pool + amount, self.energy_capacity)
    
    @property
    def energy_deviation(self):
        """归一化的能量偏离信号,供内感受使用"""
        target = self.energy_capacity * 0.5
        return (self.energy_pool - target) / target  # ∈ [-1, +1]
```

**关键: 神经系统的能量消耗是无条件的**。神经系统不能"决定"自己要不要消耗能量——它消耗是因为它在运行。能量耗尽就是肉体死亡,神经系统跟着停止。

### 4.3 PhysicalState(物理状态)

```python
@dataclass
class PhysicalState:
    position: ndarray(2,)        # 2D 坐标
    heading: float               # 朝向角(弧度)
    
    # 身体形态(分段简化)
    n_segments: int = 12
    segment_angles: ndarray(12,) # 每段相对前段的角度
    segment_positions: ndarray(12, 2)
    
    # 肌肉张力
    dorsal_tension: ndarray(12,) # 背侧肌肉张力
    ventral_tension: ndarray(12,)# 腹侧肌肉张力
    
    # 累积状态
    age_in_ticks: int
    cultivation_temp: float      # 记忆的成长温度(供 AFD 用)
```

### 4.4 SensoryTranslator(grounded 感觉转换)

**这是 grounding 实质实现的核心模块**。所有从环境/代谢到神经系统的信号都在此 grounded 化。

> 设计原则(v0.3 显式化): 所有感觉输入必须 grounded 在 set point 偏离
> 或预测信号上,不能是绝对值。换言之,送进神经系统的从来不是"温度 / 浓度 /
> 接触强度",而是"我此刻偏离能存活的状态多远 / 我下一刻的代谢负担会怎么变化"。
> 具体公式的写法将参考真实秀丽虫的电生理记录(详见 Phase 0.5 验证报告
> `PHASE0.5_REPORT.md`),实施时挑选与文献最一致的偏离/微分编码。
> 这条原则比任何具体公式都重要——一旦肉体层把绝对值喂进神经系统,
> grounding 的承诺就破了。

```python
class SensoryTranslator:
    def translate(self, env, physical, metabolism) -> ndarray(N,):
        I = np.zeros(N)
        
        # === 内感受(grounded 在代谢上)===
        # 能量偏离 → 假想的内感受神经元
        # 注: 秀丽虫真实的能量感受机制不完全清楚,这里用简化方案
        I[ENERGY_SENSOR_idx] = metabolism.energy_deviation
        
        # 温度偏离(AFD 神经元的真实功能)
        local_temp = env.temperature_field(physical.position)
        temp_dev = local_temp - physical.cultivation_temp
        I[AFD_L_idx] = temp_dev
        I[AFD_R_idx] = -temp_dev   # AFD pair 的反向对称编码
        
        # === 外感受(grounded 因为它们预测未来的内感受变化)===
        # 化学浓度的变化率(ASE 的真实编码)
        local_chem = env.chemical_field(physical.position)
        chem_change = local_chem - self._last_chem
        self._last_chem = local_chem
        I[ASE_L_idx] = chem_change      # ASEL 对增长敏感
        I[ASE_R_idx] = -chem_change     # ASER 对衰减敏感
        
        # 有害化学(ASH)
        nox = env.noxious_field(physical.position)
        I[ASH_L_idx] = nox
        I[ASH_R_idx] = nox
        
        # 机械接触
        contacts = env.detect_contacts(physical)
        I[ALM_idx] = contacts.head_touch    # 前端接触 → 触发后退
        I[PLM_idx] = contacts.tail_touch    # 后端接触 → 触发前进
        
        return I
```

**关键设计承诺**:
- 所有 sensory neuron 接收的都是**偏离信号或变化率**,不是绝对值
- AFD 的输入是"温度 - 成长温度",不是"温度"
- ASE 的输入是浓度变化,不是浓度本身
- 这一步是 grounding 的具体实现位置

### 4.5 MotorExecutor(经过肌肉物理层的运动)

```python
class MotorExecutor:
    def execute(self, V, physical):
        # === 步骤 1: 运动神经元 → 肌肉张力 ===
        # 95 个运动神经元按身体位置分组
        # A 类运动神经元 → 后退驱动
        # B 类运动神经元 → 前进驱动
        # D 类运动神经元 → 对侧抑制(GABA)
        
        for seg_idx in range(physical.n_segments):
            # 找到投射到这一段的运动神经元
            dorsal_neurons = self.DORSAL_MOTOR_MAP[seg_idx]
            ventral_neurons = self.VENTRAL_MOTOR_MAP[seg_idx]
            
            # 张力是神经元活动的低通滤波
            target_d = np.mean([V[i] for i in dorsal_neurons])
            target_v = np.mean([V[i] for i in ventral_neurons])
            
            # 一阶滞后(肌肉慢于神经)
            physical.dorsal_tension[seg_idx] += \
                (target_d - physical.dorsal_tension[seg_idx]) / MUSCLE_TAU
            physical.ventral_tension[seg_idx] += \
                (target_v - physical.ventral_tension[seg_idx]) / MUSCLE_TAU
        
        # === 步骤 2: 肌肉张力 → 身体角度 ===
        for seg_idx in range(physical.n_segments):
            target_angle = (physical.dorsal_tension[seg_idx] - 
                           physical.ventral_tension[seg_idx]) * MAX_BEND_PER_SEG
            physical.segment_angles[seg_idx] += \
                (target_angle - physical.segment_angles[seg_idx]) / BODY_TAU
        
        # === 步骤 3: 身体形态 → 位移 ===
        # 计算所有节段的位置
        self._update_segment_positions(physical)
        
        # 简化的流体动力学:
        # 正弦波传播沿身体 → 推动力
        body_wave = self._compute_body_wave(physical.segment_angles)
        propulsion = body_wave.amplitude * PROPULSION_FACTOR
        
        # 应用位移到头部,身体跟随
        physical.position += propulsion * np.array([
            np.cos(physical.heading), 
            np.sin(physical.heading)
        ])
        
        # 转向 = 身体头段角度的累积偏置
        physical.heading += physical.segment_angles[0] * TURN_FACTOR
        
        # 运动强度(用于代谢计算)
        movement_intensity = abs(propulsion) + 0.3 * abs(physical.segment_angles[0])
        
        return movement_intensity
```

**关键设计承诺**:
- 神经元活动**不直接**控制速度或方向
- 必须经过: 神经 → 肌肉张力 → 身体角度 → 形变 → 位移
- 流体动力学用简化模型(身体波形 × 系数 = 推动力),不是完整 Stokes 流体力学
- 这个简化保留了"神经必须学会协调肌肉模式"的核心物理约束

### 4.6 FoodIntake 和 DamageMonitor

```python
class FoodIntake:
    def check_and_eat(self, env, physical, metabolism):
        for patch in env.food_patches:
            d = distance(physical.position, patch.position)
            if d < CONTACT_RADIUS:
                amount = min(patch.density * INTAKE_RATE, 
                            metabolism.energy_capacity - metabolism.energy_pool)
                metabolism.eat(amount)
                patch.density -= amount * 0.5  # 食物被消耗(部分)
                return True
        return False

class DamageMonitor:
    """检测累积损伤,可能导致死亡"""
    def check(self, env, physical, metabolism):
        # 极端温度暴露
        local_temp = env.temperature_field(physical.position)
        if local_temp < TEMP_LETHAL_LOW or local_temp > TEMP_LETHAL_HIGH:
            metabolism.energy_pool -= EXTREME_TEMP_DAMAGE_RATE
        
        # 有害化学暴露(累积)
        nox = env.noxious_field(physical.position)
        if nox > NOX_LETHAL_THRESHOLD:
            metabolism.energy_pool -= nox * NOX_DAMAGE_FACTOR
        
        # 这里不直接判定死亡,通过让 energy_pool 下降间接导致死亡
        # 这样死亡的唯一原因始终是 energy_pool < 0,统一接口
```

---

## 5. 环境设计

### 5.1 物理空间

```python
class Environment:
    size: Tuple[float, float] = (60.0, 60.0)  # 60x60 单位,模拟培养皿
    
    food_patches: List[FoodPatch]
    noxious_sources: List[NoxiousSource]
    temperature_map: TemperatureField
    
    def chemical_field(self, pos) -> float:
        """位置 → 食物相关化学浓度"""
        return sum(patch.field_at(pos) for patch in self.food_patches)
    
    def noxious_field(self, pos) -> float:
        """位置 → 有害化学浓度"""
        return sum(src.field_at(pos) for src in self.noxious_sources)
    
    def temperature_field(self, pos) -> float:
        """位置 → 温度"""
        return self.temperature_map.at(pos)
    
    def detect_contacts(self, physical) -> ContactInfo:
        """检测身体与环境的机械接触"""
        # 简化:边界碰撞、食物斑块边缘接触等
        ...
    
    def step(self):
        """环境演化:食物再生、扩散等"""
        for patch in self.food_patches:
            patch.density = min(patch.density + REGEN_RATE, patch.max_density)
```

### 5.2 食物斑块

```python
class FoodPatch:
    position: ndarray(2,)
    radius: float = 2.0
    max_density: float = 1.0
    density: float                # 当前浓度
    
    def field_at(self, pos):
        d = distance(pos, self.position)
        return self.density * np.exp(-d**2 / (2 * self.radius**2))
```

### 5.3 温度场

简单梯度场,模拟培养皿上的温度斜坡:

```python
class TemperatureField:
    T_min: float = 17.0
    T_max: float = 25.0
    
    def at(self, pos):
        # 沿 x 方向的线性梯度
        return self.T_min + (self.T_max - self.T_min) * pos[0] / ENV_SIZE_X
```

### 5.4 时间结构

```python
WORLD_RATE = 10  # 环境每 10 tick 更新一次(食物再生、扩散)
```

---

## 6. 数据来源要求

### 6.1 严格要求文献支撑的必须数据

**必须**有具体文献来源,记录在 data_audit.md 中:

| 数据 | 主来源建议 | 状态 |
|---|---|---|
| W_chem 连接矩阵 | Cook et al. 2019 Nature(corrected July 2020) | 已集成(Phase 0) |
| W_gap 连接矩阵 | Cook et al. 2019 Nature(corrected July 2020) | 已集成(Phase 0) |
| 神经元递质类型(GABA 26 个) | McIntire 1993 / Schuske 2004 / Gendrel 2016 | 已集成(Phase 0) |
| 其他递质明细(ACh / Glu) | Pereira 2015 / Serrano-Saiz 2013 + WormAtlas | 待集成(Phase 1+) |
| 调质产生神经元 | Sulston 1975 / Sze 2000 / Alkema 2005 | 部分已知 |
| 可塑性神经元清单 | 多篇研究整合,见 data_audit.md | 待整合 |
| 神经元解剖位置 | WormAtlas | 现成可用 |
| 真实电生理参照数据 | Atanas 2023 eLife / Kato 2015 Cell | 待集成(Phase 0.5) |

### 6.2 允许工程假设的部分

下列内容可以有据可依的工程假设**,但必须在文档中明确标注:

- 调质对神经元兴奋性的具体影响系数(S 矩阵的非零值)
- 神经元的时间常数差异(初版统一,后期可差异化)
- 神经元的静息电位偏置
- 肌肉的具体动力学参数(MUSCLE_TAU 等)
- 流体动力学简化参数(PROPULSION_FACTOR 等)
- 学习率、衰减率、噪声水平

这些必须在代码中**必须**:
- 以常量形式集中定义,不散落
- 附注释说明"假设值,不是从生物数据来的"
- 容易调整(为后续校准留接口)

### 6.3 数据审计表的形式

data_audit.md 需要单独维护,格式:

| 数据点 | 项目中的使用位置 | 主来源 | 证据强度(1-5) | 我们的简化 | 不确定度评估 |

每个生物学声明都应该指向审计表的一行。

---

## 7. 验证策略

### 7.1 主指标:autopoietic 能力

A1: 基线生存时间
- 测量: 在标准环境中,从初始化到死亡的平均 tick 数
- 目标: 显著高于随机基线(随机运动的虫子的生存时间)

A2: 扰动恢复能力
- 测量: 受到非致命扰动(瞬时移位、临时高代谢)后,能否恢复到稳定行为
- 目标: 大部分扰动能在 10^3 tick 内恢复

A3: 个体差异
- 测量: N 只虫子的生存时间分布
- 目标: 有显著的个体差异(说明随机初始化和随机噪声真的产生不同的"虫生")

### 7.2 副指标:行为复现度

按复杂度排序的行为列表,每项给出量化方式:

B1: 基础运动
- ✓ 自发游动: 静止状态下能开始游动
- ✓ 触碰反射: 前端机械刺激 → 后退,后端 → 前进加速

B2: 趋化性
- ✓ 化学梯度攀爬: 在食物梯度中向高浓度移动
- ✓ 有害化学回避

B3: 状态依赖行为
- ✓ 饥饿增加探索: energy_deviation < 0 时活动度增加
- ✓ 温度趋向: 长期暴露后趋向 cultivation_temp

B4: 简单学习(需可塑性激活)
- 化学物质-食物关联(数小时仿真时间)

### 7.3 副指标:神经活动相关性

如果有时间,跟真实秀丽虫的钙成像数据对比:
- AVA/AVD 神经元在后退前激活
- AVB/PVC 神经元在前进时激活
- ASE 活动 ∝ 化学浓度变化

---

## 8. 实施阶段

### 阶段 0: 神经系统骨架(目标 1 周)

- 加载 Cook 2019 connectome 数据
- 构建 W_chem 和 W_gap 矩阵
- 实现 CTRNN 动力学(无调质,无可塑性)
- 单元测试:给定固定输入,系统达到稳态;移除输入,系统衰减到 0
- 验收: 神经系统可以稳定运行 10^4 tick 无数值问题

### 阶段 1: 肉体接口(1-2 周)

- 实现 Metabolism, PhysicalState
- 实现 SensoryTranslator(grounded 感觉)
- 实现 MotorExecutor(简化版,先不做肌肉物理层)
- 实现简单环境(单个食物斑块)
- 验收: 虫子在环境中能因为代谢耗尽而死亡

### 阶段 2: 完整肌肉物理层(1-2 周)

- 把 MotorExecutor 升级为带肌肉张力的物理层
- 验证身体能产生协调的游动模式
- 验收: 内部产生正弦波形的肌肉激活,虫子能定向移动

### 阶段 3: 神经调质(1-2 周)

- 实现 5 种调质的浓度动力学
- 集成 S 敏感度矩阵
- 验收: 调质对行为有明确可观测的影响(饱足/饥饿状态切换)

### 阶段 4: 可塑性激活(1-2 周)

- 加载/构建可塑性掩码 P
- 实现 Hebbian 学习
- 验收: 系统能展示简单关联学习

### 阶段 5: 系统对比研究(开放时长)

完整系统跑起来后,进行系列对比实验:
- 完整 vs 无调质 vs 无可塑性 vs 随机化 connectome
- 量化各组件的贡献度

---

## 9. 明确的设计选择

下列决策是经过明确思考后确定的,不再讨论:

| 决策 | 选择 | 理由 |
|---|---|---|
| 自我建模层 | 不加入 | 秀丽虫真实大脑可能没有,作为后续对照 |
| 备份/复活 | 严格禁止 | autopoiesis 的根基,不可妥协 |
| 衰老建模 | 初版不做 | 复杂度过高,可作后期扩展 |
| 繁殖建模 | 初版不做 | 单世代版本先跑通 |
| 流体动力学 | Level 1(简化波形传播) | 物理一致性 vs 复杂度的平衡点 |
| 跨代演化 | 暂不做 | 等单世代稳定再考虑 |
| 神经元时间常数 | 初版统一 | 数据支撑不足,后期可差异化 |
| 突触延迟 | 初版忽略 | 秀丽虫尺度上影响有限 |

---

## 10. 开放问题

下列问题在实施中需要逐步解决:

1. 调质对神经元的具体影响系数: S 矩阵的大部分位置目前没有定量数据,需要工程假设
2. 可塑性神经元的精确清单: 需要整合多篇文献
3. 肌肉到运动神经元的精确映射: 95 个运动神经元到 12 个身体节段的对应关系需要从解剖学整理
4. 流体动力学常数的校准: PROPULSION_FACTOR 等参数初版用猜测值,后期需要校准
5. 测试用环境的设计: 不同验证目标需要不同的环境配置,需要建立标准测试环境集
6. 可视化和调试工具: 需要能实时观察 302 个神经元活动 + 身体姿态 + 环境的工具
7. 数据审计表的完整版: 当前是骨架,需要逐项填充

---

## 11. 项目意义

如果这个项目能展现:

- 数字秀丽虫在生物合理时长内 autopoietic 持存
- 行为复现度达到 60%+
- 不同对照组(无调质/无可塑性等)的行为差异显著且可解释

那么:

- HSGT 框架的可证伪工程验证完成
- ALGOS 框架的第一个具体实例存在
- 关于生物智能"必要成分"的讨论从哲学转向实验
- 后续 Drift 项目(从零演化)有了清晰的参照

如果失败,差距本身就是研究产出——它精确告诉我们生物系统中哪些细节真的不能被纯逻辑捕获。

无论结果如何,这是个不会浪费的项目。

---

## 12. 与外部资源的关系

### 12.1 数据(直接采用)

下列数据来自社区,我们直接使用:

- Cook et al. 2019 connectome(已采用,Phase 0)— corrected July 2020 版本
- Atanas et al. 2023 全脑钙活动数据(Phase 0.5 验证用)
- Kato et al. 2015 全脑钙活动数据(备选 / 交叉验证)
- 标准行为测试范式与对照实验设计(Phase 1+ 验证套件)

数据可以直接用,因为数据是被独立采集的客观事实,不带方法学绑定。

### 12.2 代码(借鉴但重写)

下列工具我们看,但不直接引入依赖:

- OpenWorm Sibernetic — 软体物理仿真。Phase 2 时我们写简化版,
  保持理论可见性,避免被 SPH 实现的具体细节牵着走。
- OpenWorm c302 — 神经元模型生成。我们可以借鉴它的 tau 等参数(τ_s /
  τ_i / τ_m 的差异化策略),但不直接使用它的代码——那会让项目变成
  c302 的衍生。
- OpenWorm WormSim — 整合仿真。仅作参考。

### 12.3 为什么不直接用 OpenWorm 的全套

ALGOS-Celegans 的目的是"用通用 autopoietic 框架的最简版本实现秀丽虫",
不是"复刻秀丽虫"。直接使用 OpenWorm 组件会让项目变成 OpenWorm 的衍生,
失去独立的对照实验价值。

我们保持最简框架 + 借用数据 + 自己实现的路线,以便清晰地评估
"通用 ALGOS 框架在秀丽虫这个具体物种上能跑多远"。差距本身是产出。

### 12.4 反向参考关系

我们的发现(尤其是负面结果,例如"没有 X 的简化系统就跑不出 Y 行为")
对 OpenWorm 之类的"完整复刻"项目是补充信息——它们能从我们的剔除实验
反推哪些细节真的承担了功能。沟通保持开放。

---

*Last updated: 2026-05-20 (v0.3)*
*Status: v0.3 设计文档,主参考*
*配套文档: docs/phase0.md(阶段 0 任务书)、PHASE0_REPORT.md、PHASE0.5_REPORT.md*