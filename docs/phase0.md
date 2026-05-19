# ALGOS-Celegans · 阶段 0 实施任务书

> 此文档配套 algos_celegans_design_v0.2.md
> 任务范围: 实现神经系统骨架,可独立运行
> 预计工作量: ~1 周
> 目标受众: 实施者(包括 Claude Code)

---

## 阶段 0 范围

### 在范围内

- 项目脚手架(目录结构、依赖管理、测试框架)
- Connectome 数据加载与预处理
- CTRNN 神经动力学的纯实现(无肉体、无环境)
- 神经系统的稳定性测试
- 可视化神经元活动的简单工具

### 不在范围内

阶段 0 明确不实现:
- 肉体层(代谢、感觉、运动)— 阶段 1
- 神经调质 — 阶段 3
- 可塑性 — 阶段 4
- 环境 — 阶段 1
- 任何"行为"输出 — 此阶段是纯神经系统验证

阶段 0 的产出是一个孤立运行的神经系统,能加载 connectome、对给定输入产生稳定响应、被测试覆盖。

---

## 验收标准

阶段 0 完成的判定:

AC1 数据加载正确
- Cook et al. 2019 connectome 数据被正确解析
- W_chem 和 W_gap 矩阵的统计特征(连接数、稀疏度)与文献报告一致
- 神经元身份、解剖位置等元数据可查询

AC2 动力学稳定
- 给定零输入,系统状态收敛到稳态(衰减到 0 附近)
- 给定常数输入,系统状态收敛到非零稳态
- 给定脉冲输入,系统状态先响应后衰减
- 在 10^5 tick 的连续运行中,状态向量不出现 NaN/Inf,不出现指数爆炸

AC3 测试覆盖
- 核心模块的单元测试通过
- 至少包含:数据加载测试、动力学步骤测试、长期稳定性测试

AC4 可观测性
- 可以实时或离线绘制神经元活动
- 至少有"302 神经元 × 时间"的活动矩阵可视化

---

## 项目结构

algos-celegans/
├── pyproject.toml              # 依赖声明(用 uv 或 poetry)
├── README.md                   # 项目说明,指向主设计文档
├── data/
│   ├── connectome/             # connectome 原始数据
│   │   └── README.md           # 数据来源和下载说明
│   └── neuron_metadata.json   # 神经元元数据(名称、类型、位置等)
├── src/
│   └── algos/
│       ├── __init__.py
│       ├── config.py           # 集中的参数定义
│       ├── connectome.py       # connectome 加载和处理
│       └── neural/
│           ├── __init__.py
│           ├── dynamics.py     # CTRNN 动力学核心
│           └── state.py        # 神经状态类
│       └── viz/
│           ├── __init__.py
│           └── activity.py     # 神经活动可视化
├── tests/
│   ├── test_connectome.py
│   ├── test_dynamics.py
│   └── test_stability.py
├── notebooks/
│   └── 01_neural_dynamics_exploration.ipynb
└── scripts/
    ├── download_data.py        # 下载 connectome 数据
    └── run_basic_simulation.py # 跑基础仿真展示

---

## 技术栈

- Python: 3.11+
- 核心数值: NumPy(主要)
- 数据处理: pandas(读取 CSV/Excel)
- 可视化: matplotlib
- 测试: pytest
- 类型检查: mypy(推荐但非必需)
- 依赖管理: uv 或 poetry

性能要求: 单 tick 在 CPU 上 < 1ms。302 神经元规模的矩阵运算 NumPy 完全够用,不需要 GPU/Numba/Cython。

---

## 任务一: 数据获取与预处理

### 1.1 Connectome 数据源

主数据: Cook et al. 2019 *Nature* "Whole-animal connectomes of both Caenorhabditis elegans sexes"
- 下载地址: WormWiring (https://wormwiring.org) 提供的补充数据
- 格式: Excel 文件(.xlsx)包含化学突触矩阵和电突触矩阵
- 文件名约为 SI 5 Connectome adjacency matrices, corrected July 2020.xlsx

雌雄同体数据就够(雄虫有额外 80 个神经元,初版不需要)。

### 1.2 神经元元数据

需要为每个神经元收集:
- 名称(如 ASEL, AVAR, RIM 等)
- 解剖类别: sensory / interneuron / motor / pharyngeal / etc.
- 递质类型: cholinergic / glutamatergic / GABAergic / etc.
- 解剖位置(可选,用于可视化)

数据源:
- WormAtlas (wormatlas.org) 对每个神经元有详细页面
- 也可以使用整理好的数据集如 OpenWorm 的 c302 项目的 metadata

实施建议:
- 不要试图从头爬数据。找现成的整理版本。
- OpenWorm GitHub 仓库里有相对完整的神经元 metadata 可借用。
- 整理为一个 neuron_metadata.json 文件,key 为神经元名,value 为属性字典。

### 1.3 数据加载接口

```python
# src/algos/connectome.py

class ConnectomeData:
    n_neurons: int = 302
    
    W_chem: np.ndarray            # (302, 302)
    W_gap: np.ndarray             # (302, 302)
    
    neuron_names: List[str]       # 长度 302
    neuron_to_idx: Dict[str, int] # 反向映射
    
    # 元数据
    neurotransmitter: List[str]   # 每个神经元的递质类型
    category: List[str]           # 解剖类别
    
    @classmethod
    def load(cls, data_dir: Path) -> 'ConnectomeData':
        """从 data/ 加载完整 connectome 数据"""
        ...
    
    def get_neuron_indices_by_category(self, category: str) -> List[int]:
        """便利方法:按类别查询神经元索引"""
        ...
```

### 1.4 矩阵的兴奋/抑制赋值

W_chem 的符号由突触前神经元的递质类型决定:
- 胆碱能(cholinergic, ACh): 兴奋性 → 保留正值
- 谷氨酸能(glutamatergic, Glu): 兴奋性 → 保留正值  
- GABA 能: 抑制性 → 取负

```python
def apply_neurotransmitter_signs(W_chem, neurotransmitters):
    """根据突触前神经元的递质类型,设置 W_chem 的符号"""
    W_signed = W_chem.copy().astype(float)
    for j, nt in enumerate(neurotransmitters):
        if nt == 'GABA':
            W_signed[:, j] *= -1  # 整列(神经元 j 的所有输出)取负
        # 其他类型保持正
        # 混合递质或未知类型: 暂时保持正,记录到日志
    return W_signed
```

W_gap 总是正值(电耦合无方向性)。

### 1.5 验证数据加载正确性

数据加载后应能复现这些已知统计:
- 神经元总数: 302
- 化学突触连接数: ~7000(具体数字取决于 Cook 2019 的统计方式)
- 电突触数(对称矩阵): ~600
- 矩阵稀疏度: < 10%
- 雌雄同体特有: ~285 个神经元有连接,其余在咽神经系统

如果数字明显偏离这些,数据加载有问题。

---

## 任务二: CTRNN 动力学核心

### 2.1 状态表示

```python
# src/algos/neural/state.py

@dataclass
class NeuralState:
    V: np.ndarray              # (N,) 神经元状态,值域 [-1, 1]
    tick: int = 0              # 当前 tick 计数
    
    @classmethod
    def initialize(cls, n_neurons: int, seed: Optional[int] = None) -> 'NeuralState':
        """初始化为小随机值"""
        rng = np.random.default_rng(seed)
        return cls(V=rng.uniform(-0.01, 0.01, n_neurons))
```

### 2.2 动力学步骤

```python
# src/algos/neural/dynamics.py

@dataclass(frozen=True)
class CTRNNParams:
    tau: float = 10.0           # 时间常数(以 tick 计)
    beta: float = 5.0           # sigmoid 陡度
    noise_level: float = 0.01

def sigmoid(V: np.ndarray, beta: float) -> np.ndarray:
    """数值稳定的 sigmoid"""
    return 1.0 / (1.0 + np.exp(-beta * V))

def neural_step(
    state: NeuralState,
    connectome: ConnectomeData,
    sensory_input: np.ndarray,
    params: CTRNNParams,
    rng: np.random.Generator,
) -> NeuralState:
    """
    单 tick 的神经状态更新.
    
    Args:
        state: 当前状态
        connectome: 加载好的 connectome
        sensory_input: (N,) 外部输入,大部分位置应为 0,
                       只有感觉神经元位置有值
        params: CTRNN 参数
        rng: 随机数生成器
    
    Returns:
        新状态(新对象,不就地修改)
    """
    V = state.V
    
    # 化学突触输入: W_chem @ sigmoid(V)
    chem_input = connectome.W_chem @ sigmoid(V, params.beta)
    
    # 电突触输入: 拉普拉斯形式
    # gap_current[i] = sum_j W_gap[i,j] * (V[j] - V[i])
    #                = (W_gap @ V)[i] - V[i] * sum_j W_gap[i,j]
    gap_input = connectome.W_gap @ V - V * connectome.W_gap.sum(axis=1)
    
    # 噪声
    noise = rng.standard_normal(len(V)) * params.noise_level
    
    # CTRNN 更新
    dV = (-V + chem_input + gap_input + sensory_input + noise) / params.tau
    V_new = V + dV
    V_new = np.clip(V_new, -1.0, 1.0)
    
    return NeuralState(V=V_new, tick=state.tick + 1)
```

### 2.3 数值稳定性

潜在的数值问题:
- chem_input 爆炸: W_chem 的值可能很大(突触数量 × 多个连接累加),需要归一化
- gap_input 数值问题: 当 W_gap.sum(axis=1) 很大时,V * sum 项可能主导

建议的归一化策略:

在 ConnectomeData 加载时归一化:
- 归一化:让每个神经元的总输入量级在 O(1)
- W_chem_normalized = W_chem / max(1, np.max(np.abs(W_chem)))
- W_gap_normalized = W_gap / max(1, np.max(W_gap))

初版用简单的全局最大值归一化。后期如果出现数值问题再细化。

---

## 任务三: 测试套件

### 3.1 数据加载测试

```python
# tests/test_connectome.py

def test_connectome_loads():
    """基础加载测试"""
    conn = ConnectomeData.load(DATA_DIR)
    assert conn.n_neurons == 302
    assert conn.W_chem.shape == (302, 302)
    assert conn.W_gap.shape == (302, 302)

def test_connectome_symmetry():
    """电突触矩阵应该对称"""
    conn = ConnectomeData.load(DATA_DIR)
    assert np.allclose(conn.W_gap, conn.W_gap.T)

def test_neuron_metadata():
    """元数据完整性"""
    conn = ConnectomeData.load(DATA_DIR)
    assert len(conn.neuron_names) == 302
    assert all(name in conn.neuron_to_idx for name in conn.neuron_names)

def test_known_neurons_present():
    """已知的关键神经元应该存在"""
    conn = ConnectomeData.load(DATA_DIR)
    for name in ['ASEL', 'ASER', 'AVAL', 'AVAR', 'PLML', 'PLMR']:
        assert name in conn.neuron_to_idx, f"Missing key neuron: {name}"
```

### 3.2 动力学测试

```python
# tests/test_dynamics.py

def test_zero_input_decays():
    """无输入时,状态应该衰减到 0 附近"""
    conn = ConnectomeData.load(DATA_DIR)
    state = NeuralState.initialize(302, seed=42)
    state.V += 0.5
    
    params = CTRNNParams(noise_level=0)
    rng = np.random.default_rng(42)
    
    for _ in range(1000):
        state = neural_step(state, conn, np.zeros(302), params, rng)
    
    assert np.max(np.abs(state.V)) < 0.1

def test_constant_input_reaches_steady_state():
    """给定常数输入,状态应该收敛"""
    conn = ConnectomeData.load(DATA_DIR)
    state = NeuralState.initialize(302, seed=42)
    
    sensory = np.zeros(302)
    sensory[conn.neuron_to_idx['ASEL']] = 0.5
    
    params = CTRNNParams(noise_level=0)
    rng = np.random.default_rng(42)
    
    for _ in range(500):
        state = neural_step(state, conn, sensory, params, rng)
    
    V_500 = state.V.copy()
    
    for _ in range(100):
        state = neural_step(state, conn, sensory, params, rng)
    
    diff = np.max(np.abs(state.V - V_500))
    assert diff < 0.01, f"Not converged, max diff: {diff}"
```

### 3.3 长期稳定性测试

```python
# tests/test_stability.py

def test_no_nan_in_long_run():
    """10^5 tick 无 NaN/Inf"""
    conn = ConnectomeData.load(DATA_DIR)
    state = NeuralState.initialize(302, seed=42)
    
    params = CTRNNParams()
    rng = np.random.default_rng(42)
    
    for _ in range(100000):
        sensory = rng.standard_normal(302) * 0.1
        state = neural_step(state, conn, sensory, params, rng)
        
        assert np.all(np.isfinite(state.V)), f"NaN/Inf at tick {state.tick}"
        assert np.max(np.abs(state.V)) <= 1.0
```

---

## 任务四: 可视化

### 4.1 活动矩阵图

```python
# src/algos/viz/activity.py

def plot_activity_matrix(
    activity_history: np.ndarray,   # shape (T, N)
    neuron_names: List[str],
    title: str = "Neural Activity",
):
    """绘制神经元活动的热图"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(activity_history.T, aspect='auto', cmap='RdBu_r', 
                    vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Neuron')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return fig
```

### 4.2 演示脚本

```python
# scripts/run_basic_simulation.py

"""跑一个基础仿真,展示神经系统的稳定性和响应"""

def main():
    conn = ConnectomeData.load(DATA_DIR)
    state = NeuralState.initialize(302, seed=42)
    
    params = CTRNNParams()
    rng = np.random.default_rng(42)
    
    n_ticks = 5000
    history = np.zeros((n_ticks, 302))
    
    for t in range(n_ticks):
        sensory = np.zeros(302)
        if 1000 <= t < 2000:
            sensory[conn.neuron_to_idx['ASEL']] = 0.5
        elif 3000 <= t < 4000:
            sensory[conn.neuron_to_idx['AVAL']] = 0.5
        
        state = neural_step(state, conn, sensory, params, rng)
        history[t] = state.V
    
    fig = plot_activity_matrix(history, conn.neuron_names)
    fig.savefig('output/basic_simulation.png', dpi=150)

if __name__ == '__main__':
    main()
```

---

## 实施注意事项

### 数据获取的实际挑战

Cook 2019 数据是 Excel 文件,可能需要手动整理为 CSV/Numpy 格式。建议:
1. 用 pandas 读 Excel
2. 转换为 Numpy 矩阵
3. 缓存为 .npz 文件,加速后续加载

如果文件结构复杂,可以参考 OpenWorm 项目的解析代码作为参考(不直接依赖,只看怎么解)。

### 关于神经元名称的"标准化"

文献中神经元名称的写法不完全统一(例如 ASEL vs ASE-L vs ASEl)。在加载时建议:
- 统一为大写无连字符形式(ASEL, AVAR, RIM)
- 提供别名查询机制

### 不要过度工程

阶段 0 的目标是让神经系统跑起来,不是写一个完美的库。约束:
- 单个文件不超过 300 行
- 不写抽象基类,除非有具体的多个实现
- 不写过度灵活的配置系统,常量直接定义在 config.py
- 测试覆盖关键路径,不追求 100% 覆盖率

### 用 git 跟踪进度

- 每个任务一个 commit
- commit message 清楚说明做了什么
- 出现问题时方便回退

---

## 完成后的下一步

阶段 0 完成后,文档 `algos_celegans_design_v0.2.md` 的"阶段 1"部分接续:加入肉体层。

但在进入阶段 1 之前,强烈建议:

1. 让阶段 0 的代码自己跑几天(或后台跑大量 tick),收集长期稳定性的数据
2. 观察"无意义"的神经活动,建立对系统动力学的直觉
3. 检查阶段 0 中暴露的设计缺陷(几乎肯定有),修订主设计文档

不要急于进入阶段 1。神经系统稳定运行是后续一切的基础,在这里多花一周,后面省一个月。

---

*Last updated: 2026-05-20*
*Status: 阶段 0 任务书 v1.0*
*配套主文档: algos_celegans_design_v0.2.md*