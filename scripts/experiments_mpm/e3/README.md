# E3：方案一的接口检验

本目录包含固定材料点的位置监督接口检查，以及后续授权的零预静置 MPM 重跑和控制点最小拟合。接口检查本身不训练；refinement 只训练垫子运动，不运行 SDS、不修改原 checkpoint。

## 输入与坐标

- 第一阶段：`trained/cat_with_cushion/cat_with_cushion_izar_v100/deform/deform_3000/obj_{0,2}.pth`。
- E2：`outputs_mpmavatar_e2/chord_cat_motion/` 中的猫轨迹。
- MPM：MPMAvatar `output/mpm_visualization/16_chord_motion/chord_motion_fps30/` 中的配置与轨迹。
- 从配置读取原始输入哈希，重建垫子体积点并逐行比较，不用最近邻重排来掩盖对应错误。

实验 16 使用 `x_mpm = x_chord + particle_shift`，猫和垫子共享平移，没有额外旋转或缩放。`x_chord = (x_asset - scene_center) * scene_scale`。参考材料点取 MPM 保存的 t=0 粒子减去平移，不能改用运动开始时已经变形的粒子作为 rest 点。

## 时间与初态

CHORD 帧 j 对应 `t_mpm = 0.25 + j / 30` 秒。实验 16 保存 30 Hz 采样，0.25 秒恰好在两帧之间；检查脚本在保存时间范围内做线性插值，明确保存左右样本与系数，不外推。这是目标近似，不等于重新求解得到精确子步状态。

MPM 在 0.25 秒静置阶段受到重力；CHORD 第 0 帧查询在源码中被 detach。这可能产生无法通过运动参数拟合的首帧绝对位置残差。检验报告量化该残差，不擅自减掉静置形变或改变参考构型。

## 运行

从 CHORDCode 根目录提交已设计好的接口检查：

```bash
sbatch scripts/experiments_mpm/e3/submit_interface_check_izar.sh
```

只检查几何、时间和来源，不加载 CUDA 模型：

```bash
/scratch/izar/ydu/.conda/envs/chord0/bin/python \
  scripts/experiments_mpm/e3/check_refinement_interface.py \
  --geometry-only --output-dir /tmp/e3_geometry_unique_run
```

输出目录必须不存在，避免覆盖先前报告。

## 检查与产物

| 检查 | 判据/用途 |
|---|---|
| 原始输入与场景归一化 | 对照 source_sha256 与场景变换 |
| 固定粒子对应 | 重新生成的体积点与 MPM 初始数组逐行一致，最大坐标差 < 2e-6 |
| 猫空间/时间复播 | 根据 E2 插值重建 MPM 保存的猫位置，最大坐标差 < 3e-6 |
| checkpoint 与 E2 一致 | 重新查询猫全部 41 帧，最大坐标差 < 3e-6 |
| 垫子查询 | 全部粒子、全部帧均有限值 |
| 渲染位置接口一致 | Gaussian 中心的 query_xyz_time 与 get_xyz_rotation 结果一致 |
| 体积点覆盖 | 最近控制点距离、原始权重和、权重 floor 主导比例；无预设物理覆盖合格阈值 |
| 梯度连接 | 选一个非零帧的位置误差，以 autograd.grad 检查垫子运动参数的有限非零梯度；无 optimizer.step |
| 初态可拟合性 | 记录静置形变、frame 0 残差与 requires_grad 状态 |

- `report.json`：机器可读的检查、警告、初态偏差、基线误差和哈希。
- `aligned_interface.npz`：rest 点、粒子 ID、对齐的目标位置、时间、插值系数，完整检查还包含原 CHORD 垫子预测。

`PASS_WITH_CAVEATS` 只表示列出的接口检查通过；不证明插值误差足够小、内部插值有物理意义、梯度数值正确或能训练成功。首帧与目标采样策略明确之前，不直接开始全帧绝对位置 refinement。

## 2026-09-30 实测结果

Izar debug 作业 `3198920`，节点 i08，29 秒，COMPLETED / 0:0。报告为 `PASS_WITH_CAVEATS`。

结果目录：`trained/cat_with_cushion/outputs_mpmavatar_e3/interface_3198920/`；日志：`logs/e3_interface-3198920.out`。

| 项目 | 实测结果 |
|---|---|
| 原始来源哈希 | config 记录的 9 个文件全部匹配 |
| 4763 粒子逐行对应 | 最大坐标误差 5.96e-8 |
| 猫共享平移 + 时间复播 | 最大坐标误差 1.19e-7 |
| 猫 checkpoint 对 E2 | 全 41 帧最大误差 0 |
| 垫子查询 | 41 × 4763 × 3 全部有限值 |
| 查询/渲染位置路径 | 检查帧 0/20/40，误差均为 0 |
| 基础/附加控制点覆盖 | 两层 weight floor 主导粒子比例均为 0；不代表物理插值已验证 |
| 帧 20 的位置 MSE | 0.00293195，仅为优化前接口探针 |
| 基础/附加运动参数梯度范数 | 0.0133245 / 0.00690020，均有限非零 |
| 时间插值 | 41 帧全部在两个保存样本之间，最近样本相距 1/60 秒 |
| 首帧对 rest 的最大距离 | 1.23e-7，近似恒等形变 |
| 首帧对静置目标的最大距离 | 4.98e-4，首帧 requires_grad=False |

结论：坐标、材料点身份、查询与梯度连接足以支持下一步最小拟合设计。尚需明确首帧处理与目标时间采样策略；本次未做有限差分梯度校验、拟合、SDS 或物理准确性验证。距离均为归一化坐标。

## 后续：零预静置与最小 refinement（用户已授权）

只取消实验 16 的 `settle_delay`，保持猫轨迹、材料、粒子、边界、dt 和 30 Hz 保存频率不变。初始粒子重新置于 rest，初速度为零；不是裁剪旧轨迹。脚本 `run_no_settle_mpm.py` 从实验 16 复制并仅改输出路由、导入定位和默认静置时间，调用原 MPMAvatar/Warp 0.10.1 求解器；全部新输出放在 CHORDCode 的 E3 目录。

```bash
sbatch scripts/experiments_mpm/e3/submit_no_settle_izar.sh
```

检查 completed/finite、是否触及域边界，并查看固定视角/中央截面视频。对新轨迹重跑接口检验；要求输入对应通过、无需时间插值且首帧误差 < 1e-5，才进行位置拟合。检查脚本已改为根据实际 delay/采样判断警告，不再硬编码旧静置结果。

最小 refinement 设计：

- 固定猫、静态 Gaussian、控制点布局/半径/朝向，只优化垫子基础与附加控制点的运动张量。
- 从原 deform_3000 初始化；无 SDS、无 ARAP、无额外正则，先隔离表示能否拟合目标的问题。
- 材料点固定随机划分 80% 训练/20% 验证，seed=0；每步 1024 训练点 × 4 个随机非零帧。
- Adam，lr=0.001，800 步。每 100 步在全部时间上评估，保存验证误差最小的状态（验证集参与模型选择，不称为独立测试集）。
- 首帧不参与训练，但验证其位置没有变化。报告点的欧氏位置 RMSE（不是逐坐标 RMSE），检查标准 checkpoint 保存/加载一致性。
- 输出目标/优化前/优化后三栏**粒子几何**视频、曲线、完整物体 mesh 轨迹和单独的 obj_2_refined.pth；原 checkpoint 保持不变。
- 可运行与误差下降仅支持表示拟合能力，不证明物理正确或最终 Gaussian 视频质量提升。

```bash
sbatch scripts/experiments_mpm/e3/submit_interface_check_izar.sh --mpm-dir <新MPM目录>
sbatch scripts/experiments_mpm/e3/submit_refine_izar.sh <新接口检查目录>
```

## 零预静置与 refinement 结果（2026-09-30）

| 阶段 | 作业 / 输出目录 | 结果 |
|---|---|---|
| 零预静置 MPM | 3198923 / no_settle_3198923 | 完整、有限值、0 触边粒子；55 帧 MP4 |
| 新接口核验 | 3198924 / interface_3198924 | PASS；41 帧无需插值，首帧最大差 1.23e-7 |
| 控制点拟合 | 3198925 / refine_3198925 | 800 步中验证误差最优为第 600 步 |

以上目录均位于 `trained/cat_with_cushion/outputs_mpmavatar_e3/`。

- 训练点欧氏 RMSE：0.0780675 → 0.00625591。
- 验证点欧氏 RMSE：0.0786197 → 0.00672028，下降 91.45%。
- 首帧位置完全不变；保存后重新加载在检查帧 0/20/40 的位置误差为 0。
- 验证集用于选择最佳 checkpoint；当前没有独立测试集、跨时间泛化或最终视频质量结论。
- MPM 新轨迹仍有局部隆起与残余形变；未证明严格无穿透。此处把它作为固定运动目标，验证控制点的表示/拟合能力。
- 优化过程有波动（第 400、700 步误差升高），没有宣称收敛；未为追求更小误差额外调参。

先看 `no_settle_3198923/mpm_process.mp4`，再看 `refine_3198925/position_comparison.mp4`（左 MPM、中原 CHORD、右 refinement）。后三栏是粒子几何比较，不是 Gaussian 外观渲染。`comparison.npz` 也保存完整垫子 mesh 顶点的优化前后运动，供下一步表面/外观检查。

## sticky → separate 单变量结果（2026-09-30）

复用 `submit_no_settle_izar.sh --surface separate`；默认仍为 sticky，不改原 baseline。对照两份实际 config（排除来源哈希）确认唯一物理差别是 surface；地面摩擦仍为默认 0，其他参数保持不变。

作业 3198933 的结果在 `no_settle_3198933/`：状态有限，但 t=0.466667 s 时 814/4763 粒子触及 x=1.96 的计算域裁剪边界，触发原有 >10% 停止条件；仅 15 帧。平均位置变化约 (+0.67093,+0.000688,-0.03022)，观察到的是明显横向滑移，不是整个垫子抬离地面。

接口作业 3198934 因轨迹覆盖不足拒绝数据，refinement 3198935 未执行并已取消。没有 separate refinement 视频/checkpoint。本轮到此保留失败结果，不加摩擦、不扩大域、不调轨迹，以维持单变量范围。MPM runner 会在保存失败轨迹后正常退出，必须以 summary.completed 判断物理流程是否完成。

首次提交 3198930 因 Shell 参数转发错误在仿真开始前失败；修复后重提，旧依赖 3198931/3198932 已取消。

## separate + 高地面摩擦对照

以 separate/μ=0 的失败组为对照，仅将地面摩擦设为 μ=1.0；猫 mesh 摩擦仍为 0。新增 `--floor-friction` 以避免将两个摩擦参数混淆，默认地面摩擦仍为 0。使用原求解器的 separate + Coulomb 速度投影，不改碰撞算法。

```bash
sbatch scripts/experiments_mpm/e3/submit_no_settle_izar.sh --surface separate --floor-friction 1.0
```

先看完整性、横向位移和触边情况；目标完整且接口检查通过后，才可沿用既有位置 refinement 设置。
