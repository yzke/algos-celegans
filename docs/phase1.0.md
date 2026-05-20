# Phase 1.0 任务书: 图本位神经系统

> 范式转换——从矩阵本位到图本位
> 配套设计稿: docs/phase1_design.md
> 用户睡觉,完全自主决策,跑完为止

## 任务结构 (5个子阶段)

### Phase 1.0.1: 图数据结构
- 新建 src/algos/graph/ (node.py, edge.py, graph.py, subgraph.py)
- 用 NetworkX 作为底层 (nx.MultiDiGraph)
- 重新加载 connectome 数据到图 (302 nodes, ~4800 edges)
- 整合神经元元数据 (category, neurotransmitter, is_plastic)
- Commit: [phase1.0.1]

### Phase 1.0.2: LIF 动力学 + 事件驱动传播
- 新建 src/algos/neural_v2/dynamics.py (lif_step)
- 新建 src/algos/neural_v2/propagation.py (SignalQueue, execute_frame)
- LIF: V(t) = V(t-1)*(1-1/tau) + input; if V>threshold → spike + reset
- 事件驱动: spike → schedule future signal by delay
- 电突触: 双向膜电位拉力
- 10^4 ticks 数值稳定
- Commit: [phase1.0.2]

### Phase 1.0.3: 子图分解
- 至少8个功能回路: 后退命令/前进命令/触碰反射/化学感觉/温度/头部CPG/摄食CPG/调质
- 重叠节点同步
- 反相关测试 (关键!)
- Commit: [phase1.0.3]

### Phase 1.0.4: 学习 + 调质
- Hebbian: ΔW = η*pre*post - λ*W
- 50-100条 plastic 边
- 至少2种调质 (RID, NSM/5-HT)
- 调质慢动力学 tau_m >> neuron tau
- Commit: [phase1.0.4]

### Phase 1.0.5: 验证
- 跟 Phase 0.9 对比3个指标
- 反相关诊断
- 子图行为定性
- PHASE1.0_REPORT.md
- Commit: [phase1.0.5]

## 优先级
必须: 1.0.1, 1.0.2, 1.0.3
应该: 1.0.4, 1.0.5

## 纪律
- 决策记录到 DECISIONS.md ## [Phase 1.0.X]
- 问题记录到 QUESTIONS.md
- 阻塞写 BLOCKERS.md 并停止
- 诚实报告,不调参美化
- 每个子阶段独立 commit

## 不要
- 不要引入 PyTorch/TensorFlow
- 不要做肉体接入
- 不要放弃诚实性追求进度