# MPM E1：独立物理实验（已归档）

本阶段在 CHORDCode 中引入 MPMAvatar 的 `warp_mpm/` 求解器，使用猫与垫子的输入网格开展独立物理实验，检查加载、接触和卸载响应。工作以实验接入、可视化和诊断为主，不包含新的物理算法，也尚未接入视频监督训练或验证 video loss 到仿真参数的梯度链路。

2026-09-17：结束本阶段诊断，后续以 `14.5_single_variable/05_dx_002` 为参考起点，暂不继续追查旧故障。

## 场景建模

- 输入：`data/cat_with_cushion/scene.glb`、猫 `obj_0.glb`、垫子 `obj_2.glb`。
- 归一化：以整个场景包围盒中心平移，再按 `1.2 / 场景最长边` 等比缩放。
- 垫子：由 `cushion_volume_particles.py` 体素化表面，再将每个 XZ 列填充到统一底面，得到平底实心高度场；每个粒子体积为 `pitch³`，质量为密度乘体积。它不是直接使用表面顶点，也不代表真实垫子的内部结构已被识别。
- 猫：绕 X 轴旋转 180°，作为指定运动的刚性网格压头；本实验没有猫与垫子的双向动力学耦合。

## 后续参考参数

参考动画：
`/home/ydu/code/MPMAvatar/output/mpm_visualization/14.5_single_variable/05_dx_002/mpm_process.mp4`

选择依据是观察到提前变形减少、接触表现更合理，而非形变量更大。该观察不等同于严格无穿透或数值收敛证明。

| 参数 | 参考值 |
|---|---|
| 求解器 / 环境 | MPMAvatar `warp_mpm` / `mpmavatar`，Warp `0.10.1` |
| 材料 | `jelly` |
| 杨氏模量 / 泊松比 | `200000` / `0.2` |
| 密度 | `1000` |
| 重力 | `(0, -9.8, 0)` |
| 计算域边长 / 网格分辨率 | `2.0` / `100³` |
| 网格间距 | `0.02` |
| 粒子采样间距 / 单粒子体积 | `0.02` / `8×10⁻⁶` |
| 垫子粒子数 | `4763` |
| 时间步 | `1/96000 s ≈ 1.0417×10⁻⁵ s` |
| 接触 | 底面 `sticky`；猫 mesh collider 摩擦系数 `0` |
| 底面高度 / 初始最低粒子高度 | `0.18` / `0.19` |
| 初始间隙 / 下压行程 | `0.01` / `0.12` |
| 等待 / 下压 / 保持 / 撤回时长 | `0.25 / 1.25 / 0.25 / 0.833333 s` |
| 总时长 | `3 s` |

间隙以初始粒子范围定义，重力沉降会改变实际接触时刻。摆放和运动属于当前场景；改变坐标尺度、几何或粒子表示后，需要重新检查参数适用性，不能直接视作真实物体的 SI 标定值。

完整配置及源码哈希在参考输出目录的 `config.json` 中；长期参考见 [MPM 参考设置](../../../docs/mpm_reference_settings.md)。**本目录旧脚本的默认参数并非以上参考参数，CHORDCode 与 MPMAvatar 的同名求解器也不能默认视为相同版本。**

## 脚本与运行

本目录仿真入口均导入 CHORDCode 的 `warp_mpm/`：

| 脚本 | 用途 |
|---|---|
| `cushion_volume_particles.py` | 网格归一化、垫子体积粒子构造与域内平移 |
| `run_mpmavatar_sphere_cushion_smoke.py` | 球形网格下压并保持，记录穿透与位移 |
| `run_mpmavatar_cat_cushion_smoke.py` | 猫网格下压、保持、撤回；支持弹性与塑性材料 |
| `run_mpmavatar_cushion_impulse.py` | 局部力脉冲及自由响应 |
| `visualize_mpmavatar_sphere_cushion.py` | 球体实验三维动画 |
| `visualize_mpmavatar_cushion_impulse.py` | 脉冲实验粒子与网格动画 |
| `submit_mpmavatar_*_izar.sh` | 对应的 Slurm 提交脚本与参数扫描 |

从 CHORDCode 根目录提交旧猫实验：

```bash
sbatch scripts/experiments_mpm/e1/submit_mpmavatar_cat_cushion_smoke_izar.sh
```

提交脚本默认使用 `/scratch/izar/ydu/.conda/envs/chord0`，可通过 `CONDA_ENV_DIR` 指定环境。需要改参数或输出目录时，查看相应 Python 脚本的 `--help` 并直接传参；提交脚本不统一转发额外命令行参数。重新运行时使用新的输出目录，避免覆盖旧结果。

猫实验输出 `trajectory.npz`、`summary.json`、`cross_section.png` 和 `response_curve.png`。本目录的猫脚本不直接生成 MP4；13–15 的双视图可视化在 MPMAvatar 中实现。

## 结果索引与结论

CHORD 旧结果归档于 `trained/cat_with_cushion/outputs_mpmavatar_e1/`；提交脚本中的默认输出路径可能位于其上一级，并不自动写入归档目录。

对照实验代码位于 `/home/ydu/code/MPMAvatar/scripts/behavior_baselines/`，结果位于 `/home/ydu/code/MPMAvatar/output/mpm_visualization/`：

| 输出目录 | 实验与结论 |
|---|---|
| `11_moving_mesh` | 指定运动平板压弹性方块，作为行为基准 |
| `13_moving_cat_mesh` | 将压板替换为猫；可见压缩，但后期侧移触边，位置裁剪造成失真 |
| `14_old_cat_volume_cushion` | 复现旧体积垫子实验；猫明显进入垫子，粒子响应很小 |
| `15_volume_cushion_baseline13` | 体积垫子使用 13 的材料与数值设置，匹配旧间隙和行程；响应明显改善 |
| `14.5_single_variable` | 以 15 为基准，分别换回旧时间步、杨氏模量、密度、重力、网格间距及各段运动时长 |

14.5 中，网格间距从 `0.03125` 减小到 `0.02` 时，粒子采样间距仍为 `0.02`，其余物理设置不变。局部平均峰值下压从 `0.05479` 降至 `0.03360`（约减少 39%），同时观察到提前变形减少。

没有单个换回旧值的参数重现旧实验仅约 `0.000284` 的局部平均峰值下压，因此尚未确定旧故障的唯一根因。降低杨氏模量产生的大位移包含明显的加载前自重下沉，不能直接作为接触改善的证据。有限输出、可观看动画和较大形变均不能单独证明物理准确。

这些实验说明结果依赖材料、数值设置及接触离散；本轮没有单独隔离粒子采样间距、单粒子体积或泊松比的影响。旧 `grid_convergence` 脚本同时改变网格、粒子间距和时间步，也不能当作单变量网格收敛证明。
