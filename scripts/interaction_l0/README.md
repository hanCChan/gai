# interaction_l0：跨接收机 TX×RX 交互现象分析（L0，只推理）

研究问题、假说与判定规则见 [docs/interaction_l0/RESEARCH_PLAN.md](../../docs/interaction_l0/RESEARCH_PLAN.md)。本目录**不训练网络**，只读取冻结的 R0/R1（可选 R2-A h200）checkpoint 和既有缓存；数据通过 `receiver_r01.data` 读取，其断言阻止访问 WiSig 最终 RX、日期 3 和等化版。

| 文件 | 作用 | 依赖 |
|---|---|---|
| `link_core.py` | 二因素分解、分半交叉拟合能量、噪声校正 + PCA 的方向估计、LEACE 式斜投影擦除 | numpy |
| `physics.py` | WiSig L-STF 描述量（CFO、SNR 代理、12 音调谱形）与按源域校准的输入干预 | numpy |
| `atlas_l0.py` | L0-A 无模型物理图谱（可加性、空间/硬件归因、内部 RX 偏移） | numpy, pandas |
| `run_l0.py` | L0-B 模型测量：`--preflight`（每种结构一个作业，写 `preflight.json`）或 `--full --shard i/N`（必须有 GO 的 preflight 且代码未变）。冻结主干、9 次小头重拟合/模型 | torch, GPU |
| `merge_l0.py` | 合并全量分片，计算集成与错误重叠 | numpy, pandas |
| `summarize_l0.py` | L0-C 冻结规则，三级判定 | numpy, pandas |
| `phase_l0_preflight.sh` | 服务器第 1 步：CPU 测试 → 无模型图谱 → GPU0 上的 preflight；到此停止 | |
| `phase_l0_full.sh` | 服务器第 2 步：检查 preflight 为 GO → N 张 GPU 并行分片 → 合并 → 汇总 | |
| `test_*.py`、`smoke_*.py` | CPU 单元测试与冒烟（含阳性/阴性对照）；`smoke_run_l0_e2e.py` 用合成数据跑真实 `run_l0.py` 全流程（有 GPU 时使用 4 个分片） | numpy, pandas, torch |

```bash
bash scripts/interaction_l0/phase_l0_preflight.sh /home/keyan/code-y-r0r1 /home/keyan/l0 [/home/keyan/code-y-r2a]
# 查看 preflight.json / checks.csv / intervention_validation.csv 后：
bash scripts/interaction_l0/phase_l0_full.sh /home/keyan/code-y-r0r1 /home/keyan/l0 [/home/keyan/code-y-r2a] 4
```

状态（2026-10-19，r1）：CPU 测试全部通过；真实 `run_l0.py` 的完整流程已在 4×V100 上用合成数据跑通（见 [REVISION_r1.md](../../docs/interaction_l0/REVISION_r1.md)）；**还没有用真实 checkpoint 和真实缓存运行**。科学阈值在看到任何服务器输出之前已固定，运行后不得修改。
