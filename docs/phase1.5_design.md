# ALGOS Phase 1.5 设计稿(初版)
# 极简身体接入:给子图提供状态依赖的驱动

> 项目阶段:Phase 1.0 完成后,进入"架构期 + 行为状态"
> 终极目标:数字生命
> 当前里程碑:让 Phase 1.0 沉默的子图被激活,让调质系统
> 获得有意义的输入,让前向↔后退命令池产生真实的反相关。
>
> 设计稿版本:0.1 (Phase 1.5+ 起草)
> 状态:初版,等待批准后进入实施
>
> 上游依赖:
> - `docs/phase1_design.md` (v1.1) §15–16:Phase 1.0 验证结果 + 遗留问题
> - `data_audit.md`:所有生物学数据的 ground truth
> - `notes/subgraph_audit_part2.md`:缺失子图候选 + 桥接连接
> - `data/modulators_full.md`:10 种调质的完整数据

---

## 0. 设计原则

继承自 Phase 1.0,加强如下:

0.1 **不做完整身体仿真**。Phase 1.5 不是 OpenWorm 路径。它的目标
不是"让数字虫子像真虫子一样在培养皿里爬",而是"让 Phase 1.0 没
被激活的功能被激活"。

0.2 **抽象,跟数字生命方向兼容**。Phase 1.5 设计的接口必须能
推广到其他物种(更大的连接组、更复杂的身体)。秀丽虫的特殊性
(比如 2D 平面、卵生)可以用,但不能让接口被它绑死。

0.3 **验证导向**。Phase 1.5 完成的判据不是"身体仿真真实",
而是"Phase 1.0 列出的三个遗留问题(沉默子图、调质惰性、命令池
+0.51 反相关)有可量化的改善"。

0.4 **保留 Phase 1.0 不变**。`src/algos/graph/` 和
`src/algos/neural_v2/` 不能被替换。Phase 1.5 通过新模块
`src/algos/body/` 和 `src/algos/env/` 接入,通过桥接边连到主
神经图。这跟 design.md §13.1 "整个系统成为同一个大图的不同区域"
是一致的。

---

## 1. 系统架构

整个系统由四个相互通信的"图区域"组成:

```
[ Environment graph ]  --(sensory edges)-->  [ Neural graph (Phase 1.0) ]
                                                     |
                                                     v
                                              (motor edges)
                                                     v
[ Body graph ]    <-----   [ Actuator interface ]
                                                     |
                                                     v
                                              (proprio + energy)
                                                     v
                                              [ Neural graph ]
```

物理实现上是四个分离的 Python 模块,但概念上是一个大图。环境
→ 感觉的桥接边、运动 → 身体的桥接边、身体 → 神经的反馈边
都是普通 Edge,只是它们的 source/target 不在同一个子图。

### 1.1 模块边界

```
src/algos/env/     # 极简环境:化学梯度场 + 温度场 + 物理空间
src/algos/body/    # 极简身体:位置/朝向 + 能量预算 + 边界
src/algos/bridge/  # 三个桥接器:sensory_in, motor_out, body_to_neural
```

`src/algos/graph/` 和 `src/algos/neural_v2/` 不变。

### 1.2 数据流

每个 tick:
1. **Env step**: 化学梯度按扩散方程演化(一次小的 numpy 卷积)。
2. **Sensory bridge**: 读取 Body 在 Env 中的位置 → 计算每个化学
   感觉神经元应该感受的浓度梯度 → 转换为 sensory_input 注入
   Neural graph(在原来 GraphSimulator.step 的 sensory_input 参数
   位置)。
3. **Neural step**: GraphSimulator.step 不变。
4. **Motor bridge**: 读取 motor 类神经元的 rate 总和 → 计算速度向量。
5. **Body step**: 更新 Body 的位置、朝向、能量。检查死亡条件。
6. **Body→Neural feedback**: 把能量、姿态、最近撞墙信号转换为
   调质输入,注入到 RIC、AVK、ALA 等调质生产者(modulators 子图)。

设计原则:每一步都是无副作用的纯函数,接受当前状态返回新状态。
桥接器是粘合层,不持有自己的动力学。

---

## 2. 环境(Environment)

### 2.1 表示

二维平面网格,每个格点存储一个标量场。场的种类(最小集合):

- `food_density`: 化学吸引物浓度(对应 ASE、AWC、AWA 应感受)
- `co2_level`: CO2 浓度(对应 BAG 应感受)
- `o2_level`: O2 浓度(对应 URX、AQR、PQR 应感受)
- `temperature`: 温度场(对应 AFD 应感受)
- `repellent_density`: 化学排斥物(对应 ASH 应感受)
- `mechanical_obstacle`: 物理障碍物 0/1 二值场(对应 ALM/AVM/PLM
  应感受当撞上时)

每个场是一个 (W, H) 的 numpy 数组。每帧一次卷积平滑(模拟扩散)。

### 2.2 更新规则

每个场都按各自的时间常数演化(food 慢、CO2/O2 中、温度极慢)。
食物消耗:当 Body 在某格,该格的 food_density 减少 ε(模拟摄食)。

### 2.3 规模

W × H ≈ 100 × 100。约 60 KB 每场,6 个场共 360 KB。每帧的卷积
~1 ms,跟 Phase 1.0 的 0.07 ms/tick 量级一致。环境演化可以低频
更新(每 10 tick 一次)。

### 2.4 不做

- 不做精确流体动力学。
- 不做多个 worms 在同一环境(单虫子)。
- 不做化学反应(只扩散 + 消耗)。

---

## 3. 身体(Body)

### 3.1 状态变量

```python
class BodyState:
    position: tuple[float, float]   # 在 Env 网格中的连续坐标
    heading: float                  # 朝向,弧度
    energy: float                   # 能量储备,死亡条件
    is_alive: bool
    posture: dict                   # 姿态参数(身体弯曲度等;Phase 1.5 用极简)
    contact: dict                   # 最近一次"撞墙"信号
```

### 3.2 演化

- `position` 每 tick 更新:`v = motor_bridge(rate)`,
  `position += v * dt`。
- `heading` 每 tick 更新:`omega = head_motor_bridge(rate)`,
  `heading += omega * dt`。
- `energy` 每 tick 减少 `metabolism_rate`,在摄食时增加。
- `is_alive`:若 energy < 0,设为 False,模拟停止。

### 3.3 不做

- 不做物理仿真(无肌肉张力、无关节角度、无柔体动力学)。
- 不做精确的身体形状(把虫子视为一个有朝向的点)。
- Phase 1.5 不做生殖(不模拟卵的产出)。

### 3.4 死亡条件

- 能量耗尽
- 撞墙累计次数超过阈值(模拟物理损坏)

死亡只导致仿真停止,不会回收资源。Phase 1.5 一次实验跑一只虫子。

---

## 4. 感觉桥接(Sensory bridge)

把 Env 状态转换为 Neural graph 中感觉神经元的输入电流。

### 4.1 化学感觉

```python
def chemo_input(env, body):
    # ASEL/R 响应 NaCl 浓度的当前值
    sens_input[idx_ASEL] = scale_chemo * env.food_density.at(body.position)
    sens_input[idx_ASER] = ... (镜像但有偏置,对应已知不对称)
    
    # AWC 响应 *变化率*
    sens_input[idx_AWCL] = scale_AWC * derivative(env.food_density at body.position)
    
    # 类似 AWA, ASH, ASK
```

`env.food_density.at(position)` 用双线性插值(连续坐标 → 标量)。

### 4.2 热感觉

```python
sens_input[idx_AFDL] = scale_AFD * (env.temperature.at(body.position) - 
                                     body.preferred_temperature)
```

AFD 是"跟偏好温度的偏差"敏感的,Phase 1.5 给身体一个 preferred_T,
模拟生物的温度记忆。

### 4.3 机械感觉

ALM/AVM/PLM 在 body.contact["where"] 对应时被触发。

### 4.4 O2/CO2 感觉

URX/AQR/PQR、BAG L/R 直接读取对应的浓度场。

### 4.5 时间尺度

感觉输入跟 Neural graph 同步,每个 tick 一次。

---

## 5. 运动桥接(Motor bridge)

把 Neural graph 中运动神经元的活动转换为身体的速度向量。

### 5.1 转换公式(最简)

```python
def motor_to_velocity(neural_state, body_state):
    # 前进信号
    forward_drive = mean(neural_state.rate[forward_command_pool])
    # 后退信号
    backward_drive = mean(neural_state.rate[reversal_command_pool])
    # 净速度
    v_magnitude = scale_v * (forward_drive - backward_drive)
    # 朝向控制由头部运动 CPG 决定
    omega = scale_omega * (mean(rate[head_left]) - mean(rate[head_right]))
    return v_magnitude, omega
```

### 5.2 不做

- 不做完整的 motor neuron → muscle activation → joint torque 链条。
- 不做身体波浪的实际传播。
- 把腹索运动 (ventral_cord_motor) 子图当作"速度幅度调节器",
  其活动均值影响 v_magnitude。

---

## 6. 身体反馈桥接(Body → neural)

这是 Phase 1.5 最关键的部分:把身体的物理状态转换为调质生产者
神经元的额外输入,从而打破 Phase 1.0 的"调质惰性"。

### 6.1 能量 → 5-HT 调质

```python
# 高能量 + 食物附近 → NSM 兴奋 → 5-HT 上升
nsm_extra_input = (env.food_density.at(body.position) > food_threshold) * 
                   k_food * (body.energy / energy_max)
# 注入到 NSML/R 的 sensory_input
```

预期:在食物附近,NSM 活跃,c_5HT 升高,前向命令池阈值上升
(抑制),后退命令池间接被影响。这给 Phase 1.0 找不到的"食物
诱导减速"提供机械路径。

### 6.2 低能量 → 章鱼胺(starvation 信号)

```python
ric_extra_input = (body.energy < energy_threshold) * k_starve
# 注入 RICL/R(章鱼胺生产者)
```

预期:低能量时,RIC 活跃,c_octopamine 升高,抑制前向、激发
后退、抑制进食。这是"饥饿觅食行为"的机械路径。

### 6.3 撞墙 → ALA + RIM 激活

```python
# 持续撞墙 → ALA(应激睡眠)
ala_extra_input = body.contact["count_recent"] * k_collision
# 撞墙 → 触发 RIM(逃跑反应)
rim_extra_input = body.contact["just_happened"] * k_escape
```

预期:多次撞墙引发应激,引发 sleep_quiescence 子图(待实现);
单次撞墙触发 RIM,通过(待加入的)酪胺臂抑制 AVB → 转向反应。

### 6.4 高 CO2 → RID 激活

直接通过感觉桥接:BAG → RID 的 3+3 个连接组接触已经存在;
让 BAG 在 CO2 高时活跃,RID 自然会被驱动。这无需新桥接,
只需让感觉真正驱动。

---

## 7. 实施模块清单

按工作量从小到大排:

| 模块 | 工作量估算 | 依赖 |
|---|---|---|
| `src/algos/env/scalar_field.py` | 4-6 小时 | numpy only |
| `src/algos/body/state.py` | 2-3 小时 | dataclass |
| `src/algos/bridge/sensory.py` | 6-8 小时 | env + body + neural |
| `src/algos/bridge/motor.py` | 3-4 小时 | neural + body |
| `src/algos/bridge/feedback.py` | 6-8 小时 | body + neural (modulators) |
| `src/algos/sim/run.py` | 4-6 小时 | wiring everything |
| 新子图:`inhibitory_command_gate` | 1-2 小时 | 加入 `circuits.py` |
| 加入 RIM 酪胺调质 | 3-4 小时 | `modulators_full.md` §4 |
| 加入 dopamine 调质 | 2-3 小时 | `modulators_full.md` §2 |
| 5-HT 生产者补 VC4/5 | 1 小时 | `modulators_full.md` §1 |
| AVA/AVB ↔ 运动神经元 整流标注 | 4-6 小时 | `edge_sign_audit.md` §4 |
| 测试 + 验证脚本 | 8-12 小时 | 所有以上 |

**总估算**:50-70 小时。比 Phase 1.0 的 5-8 小时大一个数量级,
因为模块数更多、跨模块接口需要谨慎设计。

---

## 8. 验收标准

Phase 1.5 完成的判据(必须全部满足):

### 8.1 性能与稳定性
- 数值稳定:10⁴ tick 不 NaN(沿用 Phase 1.0 标准)
- 性能:整体 < 1 ms/tick(神经 0.12 + 环境 + 身体 + 桥接 = 待量化)

### 8.2 沉默子图激活
- `pharyngeal_cpg` mean rate 从 0.000 升到 > 0.01(当虫子在食物
  上时)
- `egg_laying` mean rate 从 0.000 升到 > 0.01(在 5-HT 高时)
- `ventral_cord_motor` mean rate 从 0.000 升到 > 0.05(任何时候,
  因为运动神经元应该频繁活动)

### 8.3 调质获得意义
- c_RID 不再恒为 0;至少在 CO2 升高的实验段中 c_RID > 0.1
- c_5HT 在食物附近显著高于离开食物时(差异 > 0.1)
- c_octopamine(待加入)在低能量时 > 0.2

### 8.4 命令池反相关
- forward_command ↔ reversal_command 的 Pearson r 从 +0.51 降到
  ≤ +0.10(不要求严格反相关,要求至少解除强同步)
- 真虫子在 Atanas 数据中该 r 是 -0.3 到 -0.5

### 8.5 头部对齐指标恢复
- subspace_alignment 从 Phase 1.0 的 +0.2771 至少恢复到 +0.32
- fc_similarity 从 +0.0097 至少恢复到 +0.04

### 8.6 涌现行为(定性)
- 化学梯度趋向(chemotaxis):虫子在 4000 tick 内位置中心移向
  食物
- 逃避刺激物:虫子在 4000 tick 内位置中心远离排斥物
- 撞墙反应:撞墙后短时间内出现反向运动

不要求做行为定量分析,只要看曲线就能确认即可。

---

## 9. 风险

### 9.1 调参陷阱
身体桥接引入大量新参数(scale_v, scale_chemo, k_food, ...)。
风险:为了让 §8 的判据满足,陷入手动调参。

**对策**:每个参数设一个默认值,基于 design 文档的"合理量级",
不允许"为了让指标好看而调"。如果默认值不满足判据,记录到
DECISIONS.md 说明,寻找架构原因。

### 9.2 桥接接口爆炸
Sensory + motor + feedback 三种桥接器,每种都有 6-10 个具体桥接。
风险:接口变得复杂难维护。

**对策**:把桥接定义为数据(类似 CIRCUIT_SPECS),代码通用化处理。
每个桥接是 `(source_field, target_neuron_set, transform_func)`
的元组。

### 9.3 跟 OpenWorm 趋同
身体一旦加入,模型复杂度急速上升。容易就此偏向更精确的物理模拟。

**对策**:严格遵守 §0.1。Phase 1.5 的身体是"最简能够驱动神经"
的形式,不是"模拟秀丽虫的物理身体"。

### 9.4 性能下降
环境 + 身体 + 桥接器的额外计算可能让单 tick 时间显著上升。

**对策**:环境扩散每 10 tick 一次(不需要每 tick)。桥接计算
向量化。如果总 tick 时间超过 5 ms,先 profile 再优化。

---

## 10. 跟 Phase 1.0 / Phase 0 的关系

### 10.1 保留
- 整个 Phase 1.0 的 `src/algos/graph/` 和 `src/algos/neural_v2/`
- 所有 Phase 1.0 的测试(应该全部继续通过)
- 三个验证指标的实现(Phase 0.7+)

### 10.2 扩展
- `algos.graph.circuits.CIRCUIT_SPECS` 追加 `inhibitory_command_gate`
- `algos.neural_v2.modulators` 追加 tyramine 和 dopamine 调质
- `algos.graph.loader.DEFAULT_MODULATOR_NEURONS` 扩展

### 10.3 新增
- `src/algos/env/` 整个模块
- `src/algos/body/` 整个模块
- `src/algos/bridge/` 整个模块
- `src/algos/sim/run.py` 整合的主循环

### 10.4 不动
- `PHASE0*.md`、`PHASE1.0_REPORT.md`、`PHASE1.0_FINDINGS.md`
- `data_audit.md`(只允许 append-only 更新)

---

## 11. 后续展望(给 Phase 2/3 的接口约定)

### 11.1 Phase 2: 自然演化
身体接入后,Hebbian 可塑性更有意义(因为 pre/post 活动有了真正
的相关结构)。Phase 2 可以加入结构性可塑(边的增删),实验
"通过环境压力让连接组慢慢改变"。

### 11.2 Phase 3: 跨物种
身体接口必须保持物种中立。`Body` 和 `Env` 的接口契约用通用的
"物理状态 + 标量场"语言写,不出现"秀丽虫"这种字。这样未来
能换成果蝇、斑马鱼、甚至非生物机器人。

### 11.3 离散事件 vs 连续身体
Phase 1.5 的 Body 是连续状态(浮点位置、朝向),但跟 Phase 1.0
的离散 tick 同步。如果未来要做更精确的身体仿真(亚秒级),
身体和神经可以异步,但当前不需要。

---

## 12. 决策点(开工前需要明确)

以下问题在开工前必须有结论,否则会在实施中反复:

### Q1: 环境网格规模
建议 100×100,可调。规模决定 sensor 的空间分辨率。

### Q2: 身体形状(点 vs 椭圆 vs 弯曲段)
建议 v0:单点 + 朝向。Phase 1.5+1 再考虑弯曲段。

### Q3: 整流电突触是否一次到位
建议 Phase 1.5 只加 AVA/AVB ↔ 运动 一组(最有可能影响 +0.51 的)。

### Q4: 酪胺实现:fast(化学边)vs slow(调质)?
建议两个都加(fast 通过 RIM 加几条 sign=-1 的化学边;slow 通过
调质 bank)。这样跟生物学最接近。

### Q5: 多种感觉模态的输入归一化
建议每个模态独立 scale,初始值设为"能让对应感觉神经元达到
threshold 的 30%-50% 的 sustained 输入"。后续如果实验显示某个
模态主导,再调整。

---

## 13. 完成标志

Phase 1.5 完成时,仓库应该有:

新增:
- `src/algos/env/` 整个模块 + 测试
- `src/algos/body/` 整个模块 + 测试
- `src/algos/bridge/` 整个模块 + 测试
- `src/algos/sim/run.py` 整合主循环
- `scripts/run_phase1_5_*.py` 验证脚本
- `PHASE1.5_REPORT.md` 完整报告
- `PHASE1.5_FINDINGS.md` 简洁版

修改:
- `algos.graph.circuits` 追加 `inhibitory_command_gate`
- `algos.neural_v2.modulators` 追加 tyramine, dopamine
- `algos.graph.loader.DEFAULT_MODULATOR_NEURONS` 扩展
- `data_audit.md` 追加 Phase 1.5 修订条目
- `DECISIONS.md` 追加 Phase 1.5 节
- `ROADMAP.md` 更新当前阶段

---

*设计稿版本: 0.1 (初版)*
*起源: Phase 1.5+ 数据审计 (1.5+.1 到 1.5+.4) 的发现*
*下一步: 项目作者批准后,启动 Phase 1.5 实施*
