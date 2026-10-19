# interaction_l0：跨接收机 TX×RX 交互现象分析（L0，只推理）

研究问题、假说与判定规则见 [docs/interaction_l0/RESEARCH_PLAN.md](../../docs/interaction_l0/RESEARCH_PLAN.md)。本目录**不训练网络**，只读取冻结的 R0/R1（可选 R2-A h200）checkpoint 和既有缓存；数据通过 `receiver_r01.data` 读取，其断言阻止访问 WiSig 最终 RX、日期 3 和等化版。

| 文件 | 作用 | 依赖 |
|---|---|---|
| `link_core.py` | 二因素分解、分半交叉拟合能量、噪声校正 + PCA 的方向估计、LEACE 式斜投影擦除 | numpy |
| `physics.py` | WiSig L-STF 描述量（CFO、SNR 代理、12 音调谱形）与按源域校准的输入干预 | numpy |
| `atlas_l0.py` | L0-A 无模型物理图谱（可加性、空间/硬件归因、内部 RX 偏移） | numpy, pandas |
| `run_l0.py` | L0-B 模型测量（表征统计、四臂擦除 × 冻结头/重拟合头、输入敏感度、错误物理归因、集成） | torch, 1 GPU |
| `summarize_l0.py` | L0-C 冻结规则，三级判定 | numpy, pandas |
| `phase_l0.sh` | 服务器一键运行：测试 → 图谱 → 2 模型冒烟 → 全量 → 汇总 | |
| `test_*.py`、`smoke_*.py` | CPU 单元测试与端到端冒烟（含阳性/阴性对照） | numpy, pandas |

```bash
bash scripts/interaction_l0/phase_l0.sh /home/keyan/code-y-r0r1 /home/keyan/l0 [/home/keyan/code-y-r2a]
```

状态（2026-10-19）：CPU 测试和冒烟全部通过；`run_l0.py` 中依赖 torch 的部分**尚未在真实环境运行**。阈值在看到任何服务器输出之前已经固定，运行后不得修改。
