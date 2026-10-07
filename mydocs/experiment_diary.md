# CHORD / MPM 实验日记

整理日期：2026-09-30；最近更新：2026-10-07。先看 [当前研究状态](research_state.md)，需要查过程再按日期读本页；证据定位见 [来源索引](experiment_diary_sources.md)。

## 记录范围与可信度

- 主体是本机可读的 Codex 会话（2026-08-03～09-26），涵盖 CHORDCode、相关 MPMAvatar 工作和少量 PhysGaussian/学习记录；并非账号所有设备上的完整聊天。
- 日期按每条消息的时间戳转换为 Europe/Zurich（本期 UTC+2），跨日会话拆开，避免把整个 session 都算在创建日。重复继承的原文尽量合并；重复但略有改写的总结仍需谨慎看待。
- “聊天报告完成”是历史记录，不表示本次重新跑过或全面核验。重点后期结果另外读取了实际 config/summary/log。没有记录的日期不代表没有工作。
- 只见 Git 的日期标明“提交证据”，只见文件时间的日期标明“推定”；文件修改时间可能受复制影响。提交日不等于实验执行日。
- 位移与网格值默认属于归一化仿真坐标。早期聊天中的 mm/μm 沿用当时口径，未证明是实际物体的 SI 标定。峰值指标、接触区域和时间窗口不同的实验不能直接横比。
- 初次整理仅新增文档，没有重跑 GPU 实验、切分支或修改研究代码；后续执行与结果按日期追加，区分已完成、运行中和计划。

## 主线回顾

| 阶段 | 解决的问题 | 得到什么 | 仍缺什么 |
|---|---|---|---|
| 8/3–8/6 | V100 能否跑 CHORD | 兼容环境与最小真实训练链路 | 原配置等价性、完整质量评价 |
| 8/18–8/31 | 为什么需要物理；MPM 能否响应 | P0 接触诊断、材料/冲量响应、运动学路线 | 可信的 mesh 接触 |
| 9/1–9/12 | 求解器和建模的能力边界 | 原生 demo、数值/行为对照、surface 修复 | 真实场景迁移验证 |
| 9/14–9/17 | 旧猫垫实验为什么异常 | 13/14/15/14.5 对照，冻结参考设置 | 根因未唯一确定；选择停止追查 |
| 约 9/21–9/22 | 视频学到的运动能否驱动物理 | E2 导出、16 轨迹驱动、17 正置对照 | 效果归因、接触评价、渲染/梯度闭环 |

## 按日记录

### 2026-06-24｜建立当前代码历史（仅提交证据）

提交 `eec603f`（initial commit）。这是当前 Git 历史的一个起点；没有找到当天聊天，不能据此推断具体实验。

来源：[当天证据索引](experiment_diary_sources.md#d20260624)。

### 2026-08-03｜Izar 入门、数据准备与 V100 环境摸底

学习登录节点、计算节点、Slurm 和 Apptainer；把 mentor 的容器启动方式适配到 Izar，构建并测试 CVLab 镜像。准备 CHORD 示例数据和静态 Gaussian，下载 Wan2.2-I2V-A14B 到 scratch 并建立软链接。

新建 `chord0`，实测 README 的 PyTorch 2.7.1 cu128 wheel 无法执行 V100 kernel，原因是该 wheel 缺少 sm_70。随后转向兼容组合，逐层验证依赖。当天是“建立可运行环境”，不能写成完成论文复现。详细证据见 [V100 实验日志](izar_v100_experiment_log.md)。

来源：[当天证据索引](experiment_diary_sources.md#d20260803)。

### 2026-08-04｜跑通 V100 上的真实 CHORD 最小链路

完成 BF16→FP16、FlashAttention→SDPA 和旧版 gsplat 逐帧渲染适配，补齐 Tiny VAE。聊天报告真实 checkpoint/MMGP 加载、SDS integration 和完整训练 iteration 通过；低分辨率配置 `416*240` 实际渲染为 `416×224`，41 帧、batch 1。

提交 assassin 正式 3000-iteration 作业，并用 debug 探针验证首轮；这里不能仅凭提交记录认定正式训练已完成。相关提交：`b773953`、`84d62d3`、`6ba195c`、`7b196a0`。

来源：[当天证据索引](experiment_diary_sources.md#d20260804)。

### 2026-08-06｜扩展场景运行脚本

提交 `76f17ef`：更新 assassin，并加入猫垫、人与狗、跷跷板、灯、耳机、机械臂等 Izar 脚本。当天聊天主要是取消/重启运行的请求；完成范围以提交为准，脚本存在不等于所有场景都成功跑完。

来源：[当天证据索引](experiment_diary_sources.md#d20260806)。

### 2026-08-17｜整理工作区（仅提交证据）

提交 `e7a88cd`（clear workspace）。未找到当天研究讨论或实验结论，不补写推测。

来源：[当天证据索引](experiment_diary_sources.md#d20260817)。

### 2026-08-18｜明确实验规范与研究问题

写下 [spec.md](spec.md)：一个实验对应一个可证伪假设，先机制、再最小任务、再完整 setting。梳理 CHORD 的逐场景控制点优化、Wan-SDS、ARAP 与时间约束。

形成问题意识：二维视频看起来合理，并不保证三维接触合理；后续研究需要直接检查接触、穿透和形变因果关系。

来源：[当天证据索引](experiment_diary_sources.md#d20260818)。

### 2026-08-19｜离线评估与 video + physics 方向

讨论并实现独立 `eval.py`，目标是在不加载 Wan、不重新训练的情况下读取 checkpoint、渲染固定与环绕视角。记录提醒：缺少训练期相机/配置时，不能承诺逐像素复现原视频。该代码后续提交在 `feat/eval`。

讨论用视频模型提供动作意图与粗轨迹、物理模块提供接触响应；尚未证明 MPM 能改善完整 CHORD。

来源：[当天证据索引](experiment_diary_sources.md#d20260819)。

### 2026-08-23｜P0 接触诊断、E1 自写 MPM 与首次负面结果

锁定 `cat_with_cushion`：obj_0 猫、obj_1 地面、obj_2 垫子。P0 导出 41 帧几何并量化脚部候选接触，提交 `7144ce7`。确认垫子非封闭，不能把表面点直接当作可靠体积材料。

E1 用自写 MLS-MPM、体积代理和规定运动的球做无接触/无摩擦/有摩擦对照，提交 `4a8be76`。分辨率结果不单调；延长恢复至 5 秒仍有残余压痕。后续消融显示固定底层、时间步和位置 projection 的副作用需要排查，尤其位置修正与形变梯度 F 不一致。**这一天没有足够依据进入真实 CHORD 轨迹融合。**

另完成 MPM 学习课程，原入口为 `/home/ydu/code/code_learning/course.html`；离线评估在 `feat/eval` 提交并合入 main。

来源：[当天证据索引](experiment_diary_sources.md#d20260823)。

### 2026-08-25｜引入 MPMAvatar 求解器（仅提交证据）

提交 `e81877e`，把 MPMAvatar 的 `warp_mpm` 复制到 CHORD。它是引入代码的日期，不等于验证完成日期。

来源：[当天证据索引](experiment_diary_sources.md#d20260825)。

### 2026-08-26｜选择原生求解器、体积化与联合细化

比较自写求解器、PhysGaussian 和 MPMAvatar；研究 PhysGaussian 如何用 MPM 更新 Gaussian 位置与协方差。后续主实验选择有运动 mesh collider 的 MPMAvatar `warp_mpm`，适配 Torch/Warp 接口，并整理 P0/E1 脚本。

垫子用 height-field 列填充构造传统体积粒子；发现该版本 `neo-hookean` 选项没有有效 stress 路径，改用实际实现的 `jelly`。球体 smoke 可运行，但仍有明显侵入。

网格、粒子、时间步联合细化后异常迁移消失，形变量初步趋稳；四档最大粒子中心侵入约 14.98/13.03/15.60/11.45 mm，**没有证明接触收敛**。随后准备 impulse 实验，把材料响应与碰撞链路分开。

注意：提交 `57e5a2e` 标题写 Warp 1.61.0，而聊天记录写 1.16.0；不能把提交标题当作准确的环境锁定文件。

来源：[当天证据索引](experiment_diary_sources.md#d20260826)。

### 2026-08-29｜定位“施力后几乎不动”

初始冲量视频几乎静止。阶梯力与逐步统计显示力已写入速度，但 P2G/G2P 衰减很强，15 ms 输出间隔还漏掉快速响应。

将采样提高到每 0.2 ms，`0.1 N × 2 ms` 报告峰值位移 6.32 μm，约在 2 ms 达峰。结论是“力有效、响应小且曾被漏采样”，不能据低帧率视频否定求解器。

来源：[当天证据索引](experiment_diary_sources.md#d20260829)。

### 2026-08-30｜冲量消融和慢放可视化

比较增大力、延长施力、降低模量：同一轮消融基线 11.6 μm，10 倍力约 116.2 μm，延长施力约 16.0 μm，降低模量约 35.2 μm。同总冲量不意味着相同峰值位移。

完成 301 帧高频慢放。当天基线数值与 8/29 不同，未逐项重建两次配置，不能把两天结果当成严格同一基线直接比较。

来源：[当天证据索引](experiment_diary_sources.md#d20260830)。

### 2026-08-31｜确认运动学路线，分离冲量与接触问题

更强/更宽施力得到可见的毫米级位移，检查并修正严格边界判断排除最顶层粒子的问题。复查 baseline，聊天报告 13 帧负有向间隙、最大约 8 mm；由于网格开放，这是局部穿越代理指标，不是严格体积内外判定。

确定 [运动学命题](0831_mpm_Kinematics.md)：保持猫的主动轨迹，用 MPM 预测垫子响应。冲量实验支持材料响应存在，不能替代 mesh 接触验证。

另实现动态 mesh SDF 与网格恢复速度的探索支线，未加粒子投影；该支线后来保存到 `feat/sdf`。不能把 SDF 支线当作已合并的默认实现。

来源：[当天证据索引](experiment_diary_sources.md#d20260831)。

### 2026-09-01｜刚性猫压垫子、布料对照与 SDF 分支

SDF 改动保存为 `b7a5c03`（feat/sdf），切回 dev_mpm。讨论传统体积粒子与各向异性布料分支的区别。

新增刚性猫按规定轨迹下压/保持/撤回的实验（`63dca43`），尝试倒置猫；结果虽然有限值、能回弹，但压缩极小，不能视为接触成功。

E1.1 布料落盒实验暴露初始化错误：法向列误乘厚度、盒面三角形过稀，接触前就失稳。属于无效结果，不支持材料或碰撞结论。

来源：[当天证据索引](experiment_diary_sources.md#d20260901)。

### 2026-09-02｜识别提前接触与布料修复后的失败

猫实验完成进一步运行，但峰值接触附近仍存在几何间隙；网格尺度 0.02 的接触支撑域可能在表面接触前就驱动粒子。对齐整个猫包围盒也不保证最低接触部位对准垫子。

修正版布料作业完成，结果仍坍缩到边界。**作业 COMPLETED 与物理结果合格必须分开记录。**

来源：[当天证据索引](experiment_diary_sources.md#d20260902)。

### 2026-09-03｜MPMAvatar 独立环境、数据和 CLI

在 MPMAvatar 配置 Python 3.10 / PyTorch 2.0.0 cu118 / Warp 0.10.1 环境，GPU smoke 通过。处理 demo 下载限流、手动 ZIP 合并及模型依赖；当天仍有 Blender/FFmpeg 链路待验证。

开始用 `mpm_cli.py` 理解粒子、质量、网格和单步推进。同时检查 scratch 中环境的保存风险，准备持久备份与 base 迁移。

来源：[当天证据索引](experiment_diary_sources.md#d20260903)。

### 2026-09-04｜官方 demo 最小链路与环境保存

MPMAvatar 四帧完整链路通过：MPM→Blender AO→Gaussian 渲染→FFmpeg；作业 `3140971`，报告约 28.8 秒。完整 demo `3140973` 启动；所读聊天最后仅报告运行中，不能据此认定全程成功。

校验 chord0 压缩备份，将新 base 安装到 home；关键运行环境仍在 scratch，备份与运行副本分开管理。

来源：[当天证据索引](experiment_diary_sources.md#d20260904)。

### 2026-09-07｜恢复 chord0，探索 mesh→Gaussian 接口

发现 scratch 中 chord0 部分文件消失，从 home 备份重建并验证 Python/NumPy/PyTorch。

对比训练拟合与 mesh 直接转换两条 Gaussian 路径，并检查 MPMAvatar 的三角面绑定方式。完成直接转换与多视角渲染，为未来物理几何驱动外观探索接口；不是当天已经实现完整绑定融合。

来源：[当天证据索引](experiment_diary_sources.md#d20260907)。

### 2026-09-08｜预处理支线：UV、尺度与表面朝向

对比直接转换与优化 Gaussian，定位棋盘格纹理丢失：先取顶点颜色再插值，无法恢复原纹理细节。保留/插值 UV 并重新采样纹理后做对比。

测试法线对齐、法向薄化及切向尺度 ×0.65。聊天观察：仅对齐改善有限，缩小切向尺度减轻垫子轮廓鼓包，但出现条纹/覆盖不足，不能宣称已解决外观问题。

讨论 MPMAvatar 的 tracking、appearance、物理参数拟合三阶段。该预处理方向与接触主线应分别归档；后续 fix/penetrate 曾单独回滚相关提交。

来源：[当天证据索引](experiment_diary_sources.md#d20260908)。

### 2026-09-10｜MPM 独立过程可视化

在 MPMAvatar 用 debug 跑默认 1000 粒子、90 帧可视化，作业 `3145218`；生成 MP4、GIF 和诊断图。建立后续行为验证可复用的观察入口。

来源：[当天证据索引](experiment_diary_sources.md#d20260910)。

### 2026-09-11｜材料、时间步和粒子密度的基础实验

围绕 CLI 学习状态初始化、质量/体积、材料、重力和地面；验证无重力近似静止。对齐过 CLI 与可视化初始化，之后恢复原可视化场景参数并保留 material 参数。

原设置 sand 可跑，jelly 在大时间步下发散；只将每帧子步从 20 增至 40/100 即可消除该次 NaN。随后做 400/800/1600/3200 子步对照和体积采样实验：有收敛趋势，但未严格收敛。画面“碎开”不能自动解读为材料断裂。

同时核查猫和垫子的开放网格，意识到几何体积化是独立建模假设。

来源：[当天证据索引](experiment_diary_sources.md#d20260911)。

### 2026-09-12｜六组行为 baseline 与接触实现修复

整理编号目录，开展 06 接触模式、07 整体平移、08 加载撤去、09 网格平面、10 曲面、11 运动压板。工作基准采用 jelly / n3375 / sub3200 / 30 FPS。

06 揭示 surface separate/slip 的投影速度被错误写零；修复提交 `3b6a2bf`，行为实验提交 `ac0b76c`。该修复发生在 MPMAvatar，不能假定 CHORD 同名副本同步包含它。

决定停止全面完善求解器：已有证据足以开始接入诊断，把目标重新集中到“MPM 能为视频监督补什么”。

来源：[当天证据索引](experiment_diary_sources.md#d20260912)。

### 2026-09-14｜回到 CHORD，做 12/13/14 对照

写下 [CHORD_HANDOFF](/home/ydu/code/MPMAvatar/docs/CHORD_HANDOFF.md)，把 CHORD 作为主工作目录，MPMAvatar 作为参考实验场。

12：运动曲面完成，有卸载回弹。13：把运动压板换成倒置猫，方块明显受压，但后期横移触碰计算域并被裁剪成薄片，不能当作完全正确的 baseline。14：使用旧垫子与旧设置，仅换成统一可视化；新旧轨迹差约 1.31e-6，确认旧异常不是画法造成。

形成下一步：保持较可靠设置，替换体积垫子，区分几何假设和数值/材料组合。

来源：[当天证据索引](experiment_diary_sources.md#d20260914)。

### 2026-09-15｜15 改善响应，14.5 单变量排查

15 使用同样体积化垫子并采用 13 的材料/密度/时间步等设置，明显压缩与恢复；局部平均峰值下压从旧 14 的 0.000284 增至 0.05479。由于同时换了多项设置，不能单独归因于 dt。

14.5 做 1 个基准 + 9 个单变量回退。没有一个单项重现旧 14 的近乎无响应。dx=0.02 组峰值约 0.03360；单独改 dt 为 0.0002 仍有明显压缩。**旧故障根因仍未确定，体积化必然失败和时间步单独致错两种解释均证据不足。**

结果入口：`/home/ydu/code/MPMAvatar/output/mpm_visualization/14.5_single_variable/`。

来源：[当天证据索引](experiment_diary_sources.md#d20260915)。

### 2026-09-17｜冻结参考设置，停止追旧故障，规划 E2

你选择 14.5/05_dx_002 作为后续起点，理由是提前变形减少、接触视觉更合理，而非追求更大形变。参考：jelly，E=200000，ρ=1000，ν=0.2，域 2.0，网格 100³，pitch=dx=0.02，dt=1/96000 s。

整理 fix/penetrate 分支、记录参考参数并回滚无关预处理提交；讨论清理旧求解器和补齐 E1 README。这些操作与其最终归档提交日期分开记录。

规划正置猫下压、CHORD 猫 mesh 轨迹导出和单向驱动垫子；确认对象可分开，控制点可直接变形原 mesh，需统一坐标和时间。自由落体接触后若要求猫减速/回弹，仍需双向动力学，不能只换抛物线轨迹。当天这部分聊天停在方案和代码核查。

来源：[当天证据索引](experiment_diary_sources.md#d20260917)。

### 2026-09-21｜归档 E1；E2 导出与实验 16 已有产物

**证据不同于前几天：归档有准确 Git 日期；E2 没有找到对应聊天，日期由日志/文件修改时间推定。**

`8eb60f6` 清理旧求解器及依赖脚本，`2b76ea5` 归档 E1 和参考设置，位于 fix/penetrate。

当前磁盘的 E2 导出读取 CHORD deform_3000，输出 41 帧猫 mesh（4356 顶点、7610 面）、控制点、归一化元数据、视频和统计；`trajectory_finite=true`。导出日志为 `logs/e2_cat_motion-3185662.out`。

MPMAvatar 实验 16 用导出的 CHORD 猫轨迹驱动 mesh collider，采用 05_dx_002 数值设置；结果 JSON 显示 completed/finite=true，边界粒子峰值数 0，局部平均峰值下压 0.0158404，末帧 0.00953044。日志 `behavior-16-3185671.out` 有完成标记。**这证明已有离线轨迹驱动实验，不证明优于 video-only，也不证明视频梯度穿过 MPM。**

[导出统计](../trained/cat_with_cushion/outputs_mpmavatar_e2/chord_cat_motion/motion_stats.json) · [16 结果](/home/ydu/code/MPMAvatar/output/mpm_visualization/16_chord_motion/chord_motion_fps30/summary.json)

来源：[当天证据索引](experiment_diary_sources.md#d20260921)。

### 2026-09-22｜实验 17：正置猫下压对照（日期推定）

未找到当天聊天；产物修改时间为当天，不能据此确定精确开跑时间。README 表明该实验用正置猫做刚性竖直下压，作为 CHORD 轨迹实验的对照。

现存 summary 显示 completed/finite=true、边界粒子峰值数 0；局部平均峰值下压 0.0208774，末帧约 -8.90e-5。初始间隙 0.01，下压行程 0.12。

16 和 17 的局部选择、轨迹与时长需要对齐后才能比较，不能仅凭各自峰值判断谁更物理。

[17 说明](/home/ydu/code/MPMAvatar/output/mpm_visualization/17_upright_press/upright_press_dx002/README.md) · [17 结果](/home/ydu/code/MPMAvatar/output/mpm_visualization/17_upright_press/upright_press_dx002/summary.json)

来源：[当天证据索引](experiment_diary_sources.md#d20260922)。

### 2026-09-26｜重新浏览仓库

聊天重新介绍 CHORD 的输入、控制点、视频监督和 MPM 实验线，没有新实验记录。该次回答引用 E1 归档“尚未接入视频监督训练”，应与已经存在的 E2 离线轨迹驱动区别理解。

来源：[当天证据索引](experiment_diary_sources.md#d20260926)。

### 2026-09-30｜恢复实验记忆与当前状态核对

本次从聊天、所有本地分支和实际产物整理日记。当前分支 dev_mpm，HEAD=8b3d3ef；fix/penetrate 的归档没有体现在当前检出文件树中。已有 dynamic_gaussian_model.py 注释修改和未跟踪 e2 目录，均保留。

恢复出的实际进度是：已选参考设置、已有 E2 离线轨迹驱动和正置猫对照，下一步应先审阅这两组结果与接口，而不是重新从零做球压垫子。

来源：[当天证据索引](experiment_diary_sources.md#d20260930)。

#### 当日续记｜设计 MPM 信号进入视频优化的通路

- **目的**：以猫与垫子为例，明确物理信号、优化变量和反馈路径，再设计最小实验。
- **讨论**：方案 1 为已有 CHORD 结果的二阶段物理轨迹监督，用户认为可做，但须解决坐标/时间尺度及体积粒子与垫子运动表示的对应；方案 2 在早期抖动猫轨迹下交替刷新 MPM 目标可能放大不稳定，暂不优先；方案 3 应先有视频生成的猫运动，再生成物理垫子运动，最后进行视频 refinement。
- **方向**：用户希望重点探索方案 4，即视频梯度穿过物理仿真。mesh–Gaussian 绑定可用于构建几何梯度路径，但不会自动分离“外观梯度/位置梯度”；具体冻结、绑定和可训练参数仍需设计。
- **待验证**：可考虑在参考粒子初始位置查询 CHORD 形变场建立固定对应；内部形变查询的覆盖/权重是否合适尚未验证。可微 MPM 的输入、接触、时间推进及跨框架梯度尚未核查，不能当作已实现。
- **状态/下一步**：今天目前只有设计讨论，未改实验代码、未提交作业。先逐项确认方案及最小梯度验证的目标，不默认启动完整训练。来源为当前会话。

#### 当日实验｜方案一 E3 接口检验

- **目的**：验证实验 16 的 MPM 粒子是否可作为 CHORD 垫子控制点的位置监督，先排除坐标、时间和身份错配。
- **执行**：新增 `scripts/experiments_mpm/e3/check_refinement_interface.py`、debug 提交脚本和 README。作业 `3198920`，i08，29 秒，COMPLETED / 0:0；未创建优化器或执行参数更新。
- **观察**：4763 个粒子逐行重建误差 5.96e-8；共享平移后的猫时序复播误差 1.19e-7；当前猫 checkpoint 与 E2 全 41 帧完全一致。垫子 41 帧体积点查询均有限，两层权重无 floor 主导点；帧 20 损失到基础/附加控制点运动参数均有有限非零梯度。
- **限制**：MPM 有 0.25 s 预静置，所有 CHORD 帧的目标时刻位于两个 MPM 保存样本之间（最近相距 1/60 s），当前只提供线性插值近似。CHORD 第 0 帧 detach，其对静置后目标的最大残差 4.98e-4（归一化坐标）无法通过当前运动参数更新消除。
- **判断/下一步**：接口 `PASS_WITH_CAVEATS`，不等于训练成功或物理正确；下一步先决定首帧是否排除及目标时间采样策略，再做无 SDS 的控制点最小拟合。未擅自减去静置形变、重跑 MPM 或修改首帧逻辑。

[实验说明](../scripts/experiments_mpm/e3/README.md) · [结构化报告](../trained/cat_with_cushion/outputs_mpmavatar_e3/interface_3198920/report.json)。

#### 当日续实验｜取消静置，完成位置监督 refinement

- **用户决定**：不裁剪旧轨迹，取消 0.25 s 预静置，从 rest/零初速度重新跑 MPM；可行则继续 refine。
- **MPM**：作业 3198923，只改变静置时间，生成 55 帧视频；全部状态有限、没有触边，局部平均峰值下压约 0.0170469。查看下压帧与跨时间截图，未见整体塌缩，但仍有局部隆起/残余形变，不能将目标当作物理真值。
- **接口**：作业 3198924，41 个 CHORD 时刻全部落在实际保存的 MPM 时刻，插值帧数 0；首帧最大误差 1.23e-7，PASS。
- **refinement**：作业 3198925；固定猫与 Gaussian/控制点布局，仅优化垫子两层运动张量；80% 材料点训练，20% 验证，seed=0，Adam lr=0.001，800 步，无 SDS/额外正则。验证最优为第 600 步，欧氏位置 RMSE 0.0786197 → 0.00672028（下降 91.45%），训练 RMSE 0.0780675 → 0.00625591。首帧未改变，标准 checkpoint 重载检查误差为 0。
- **结论/限制**：初步支持现有控制点场可以吸收 MPM 位置监督；不证明最终 Gaussian 渲染改善、严格物理正确或优化收敛。误差在部分检查步反弹，保留验证最优；未扩大实验。下一步先看几何对比与表面运动，再讨论是否加入视频监督。

[MPM 视频](../trained/cat_with_cushion/outputs_mpmavatar_e3/no_settle_3198923/mpm_process.mp4) · [拟合统计](../trained/cat_with_cushion/outputs_mpmavatar_e3/refine_3198925/summary.json) · [三栏几何对比](../trained/cat_with_cushion/outputs_mpmavatar_e3/refine_3198925/position_comparison.mp4)。

#### 当日静态检查｜垫子是否被约束不离地

检查 E3 实际调用的 MPMAvatar 求解器：底面 y=0.18 使用 sticky，重力为 (0,-9.8,0)，初始底层粒子中心 y≈0.19。sticky 在地面下方网格节点把三向速度全部置零，不区分向下穿入和向上分离；因此是无滑移/黏性速度边界，会抑制近底部运动，不能当作可自由分离的单边接触。但没有将底层粒子位置固定回初值，粒子仍经 G2P 插值与位置积分更新，所以不严格保证整个垫子永不离地。计算域位置裁剪也不是底面锚定。refinement 只有粒子位置 MSE，没有额外底面或固定底层约束，仅间接继承目标轨迹的贴地倾向。此次仅静态检查、记录，没有修改求解器或重跑实验。

#### 当日单变量实验｜底面仅改 separate

用户要求仅将 sticky 改为 separate，重跑 refinement。新增 surface 参数并复用既有提交脚本；两份实际 config 核对，唯一物理设置变化为 surface，地面默认摩擦仍为 0。首次 3198930 Shell 转发错误在仿真前退出，已修复重提，取消其依赖。

MPM 3198933 数值有限，但在 0.466667 s 有 814/4763 粒子触及 x=1.96 边界而提前停止；15 帧视频完整可读。平均位置变化 (+0.67093,+0.000688,-0.03022)，主要现象为垫子横向滑走、撞域边界，而非整体离地。接口 3198934 因时长不足失败，refinement 3198935 未运行并取消。没有生成 separate refined checkpoint。

结论：当前设置明显依赖 sticky 的水平约束；仅释放为无摩擦 separate 后，不能沿用完整目标做 refinement。保留失败结果，未扩大域或另加摩擦。下一步需讨论想模拟可滑动垫子还是固定支撑，而非把本次结果当作数值 NaN 失稳。

[失败前视频](../trained/cat_with_cushion/outputs_mpmavatar_e3/no_settle_3198933/mpm_process.mp4) · [记录](../trained/cat_with_cushion/outputs_mpmavatar_e3/no_settle_3198933/README.md)。


### 2026-10-01｜扩展 demo：资产转换、五场景构建与静态预处理

**目的**：扩展猫与垫子之外的 demo；每组明确区分主动施力物体与被动受力物体，被动体后续由拉格朗日粒子仿真推演。本轮完成输入数据准备并启动静态 Gaussian 拟合，尚未验证新场景的运动或物理响应。

#### 资产准备与场景选择

- 检查 `mydata/3Dassets/` 的 27 个 ZIP：均为 Blender 场景及贴图，没有现成 OBJ。创建独立环境 `/scratch/izar/ydu/.conda/envs/chord-mesh`（Python 3.11、bpy 4.5.3），全部导出为内嵌纹理的 GLB 和配套 MTL/贴图的 OBJ。
- 转换中排除隐藏 LOD 副本与无关环境光引用；对复杂颜色混合材质显式连接源 diffuse 图，修复棒球、蛋糕遮罩被误当颜色图及部分 OBJ 缺少颜色贴图的问题。不能据此声称保留了 Blender 所有着色效果。
- 27 项通过几何与贴图检查；16 项在焊接接缝后的副本上封闭。封闭性不等于已验证体积填充、自交、本构或仿真稳定性。
- 用户选定以下五组；从多实例资产中选取一只抱枕和一个平放堆肥袋。抱枕按主轴平放，以实际顶点确定落地高度，修复初始倾斜及悬空。

| 组合 | 场景目录（`data/` 下） | 目标动作 | 预处理作业 |
|---|---|---|---|
| 27 木板 + 10 牛角包 | `cutting_board_with_croissant` | 木板倾斜压扁面包后抬起，保留压痕 | `3200136_0` |
| 05 棒球 + 22 抱枕 | `baseball_with_pillow` | 球压入抱枕再移开，抱枕回弹 | `3200136_1` |
| 19 轮胎 + 09 堆肥袋 | `tyre_with_compost_bag` | 轮胎滚过袋体，产生压缩和侧向鼓起 | `3200136_2` |
| 27 木板 + 20 橡皮鸭 | `cutting_board_with_rubber_duck` | 木板挤压橡皮鸭后松开，鸭身恢复 | `3200136_3` |
| 26 扫帚 + 23 垃圾袋 | `broom_with_trashbag` | 扫帚推动袋体沿地面移动并局部凹陷 | `3200136_4` |

#### 输入组织与监督类型核对

各场景使用 `obj_0.glb` 主动体、`obj_1.glb` 静态地面、`obj_2.glb` 被动体；`scene.glb` 仅包含主动体与被动体，作为共同归一化参考。另附 `prompt.txt`、`negative_prompt.txt`、双视角静态预览、放置元数据和校验报告。五组均通过共享坐标、纹理内嵌、落地高度及初始包围盒分离检查；原预处理入口 dry-run 正确识别三对象。主动/被动标签仅为场景约定，不会自动给 CHORD 加上物理约束。

原有英文 prompt 位于 `example/*.sh` 的 `--prompt` 参数；新 prompt 仿照“主动动作 + 被动接触形变”编写，场景文本需显式传入参数。用户随后指定只仿照 `example/assassin_with_eagle_izar_v100.sh` 创建提交脚本，已保持其资源及训练参数（包括 `--add_cp_objs obj_2`）。**该模板实际是 `--task i2v-A14B` / WanI2V，使用 Wan2.2-I2V-A14B，并非先前口头计划中的 TI2V-5B。** 五份运动训练脚本已写好，尚未提交。

#### 静态 Gaussian 预处理：已启动，未完成

新增 `example/preprocess_new_demos_izar_v100.sh`，通过 Slurm 数组 `3200136` 提交五场景，每个申请 1 GPU、8 CPU、64 GB 内存、30 小时上限。复用原有 `scripts/preprocess/train_gaussian_for_scene.py`，按 `obj_0 → obj_2 → obj_1` 处理，未重写 Gaussian 拟合逻辑。

本次更新时 `sacct` 显示五个子任务均为 **RUNNING**，已运行约 6 分钟；实际日志显示均在首个对象 `obj_0` 的 50,000 步拟合中，约 5,700～6,730 步。这里记录的是运行快照，不能算作预处理成功。输出位于 `trained/<场景名>/`，日志为 `logs/preprocess_demos-3200136_<0–4>.out`。

#### 工具归档与下一步

按用户要求，将资产工具统一放入 `scripts/preprocess/blender2glb/`，包括转换、场景组装、校验脚本及环境 YAML；修正根目录定位和文档路径，迁移后五场景检查与脚本语法检查通过。`dev_mpm` 分支分为三个本地提交，尚未 push：

- `8a08709`：Blender 资产转换与场景准备工具。
- `095f6ae`：五场景 Gaussian 预处理 Slurm 数组。
- `64bf42f`：五场景 I2V-A14B 运动训练脚本。

生成资产、场景、日志和训练输出仍按 `.gitignore` 留在本地。下一步等待预处理结束，检查各对象 checkpoint 与静态 Gaussian 渲染，再决定启动运动训练。本轮未执行新场景 Wan 监督或 MPM 仿真；袋体、橡皮鸭的体积/材料近似和扫帚薄片刷毛的碰撞代理仍需另行设计。

来源：当前会话及本次实际读取的 `sacct`、日志、Git 提交；[资产清单](../mydata/3Dassets/ASSET_CATALOG.md) · [场景检查页](../data/review_new_demos.html) · [工具说明](../scripts/preprocess/blender2glb/README.md) · [预处理提交脚本](../example/preprocess_new_demos_izar_v100.sh)。


### 2026-10-02｜五场景 preprocess 完成；切换 TI2V，debug 后提交纯 CHORD

- **预处理**：数组 `3200136_0–4` 全部 COMPLETED / 0:0，各耗时约 2 小时 18 分～2 小时 28 分。15 个对象的最终 `iteration_50000/point_cloud.ply` 均存在，日志有逐场景完成标记。此前等待 20 分钟后的检查仍未结束，后续会话继续时核实全部完成。
- **监督切换**：按用户新指示从 I2V-A14B 改为 `ti2v-5B`，下载官方 Wan2.2-TI2V-5B 权重到 `/scratch/izar/ydu/checkpoints/Wan2.2-TI2V-5B`。五个脚本直接读取之前各场景的 `prompt.txt`；41 帧、832×480、batch size 1，完整训练 3000 步。未接入 MPM、物理轨迹或 refinement。
- **兼容修改**：TI2V 在 V100 上使用 FP16 DiT；保留完整 Wan2.2 VAE、CPU T5，移除 I2V 的 Tiny VAE/MMGP 参数。补齐 TI2V latent-shape 与分段 SDS target/loss 接口，复用已有 SDS 目标计算和首帧梯度排除；DiT 生成目标后卸载至 CPU，再进行 VAE 与七帧一组的 render backward。修复 SDS setup 强制放大到 704×1280 的问题，按请求尺寸计算 render/latent 网格。
- **先跑扫帚**：首次 debug `3200310` 加载权重后因缺少 `_compute_latent_shape` 接口失败，未进入优化。修复后 `3200311` 完成 3 步，COMPLETED / 0:0，耗时 4:36。两个动态对象 checkpoint 均有限，第 1→3 步运动参数最大变化分别约 0.01196，确认真实反传和参数更新。
- **按顺序扩展**：扫帚先提交完整作业 `3200312`，之后其余四个 debug `3200313–16` 均 COMPLETED / 0:0，约 4:44～5:12；四组 `obj_0/obj_2` checkpoint 全部有限且有非零更新。3 步 smoke 仅证明初始化、SDS/反传及保存路径可运行，不证明运动质量、收敛或第 300 步附加控制点阶段已验证。
- **时限修正**：debug 实测约 35～44 秒/步（包含频繁保存），完整 3000 步可能超过原 30 小时。normal QoS 最大允许 72 小时，完整脚本改为 48 小时。集群拒绝延长运行中的 `3200312`，因此将其取消，保留已有输出，并以独立 `_48h` 目录先重提扫帚 `3200320`；再提交其余四个完整作业。完整实验均从静态 Gaussian 重新初始化，没有接着 debug checkpoint 训练。

| 场景 | 通过的 debug | 正式作业 | 正式输出目录（`trained/<场景>/` 下） |
|---|---|---|---|
| `broom_with_trashbag` | `3200311` | `3200320` | `broom_with_trashbag_ti2v_izar_v100_48h` |
| `cutting_board_with_croissant` | `3200313` | `3200321` | `cutting_board_with_croissant_ti2v_izar_v100` |
| `baseball_with_pillow` | `3200314` | `3200322` | `baseball_with_pillow_ti2v_izar_v100` |
| `tyre_with_compost_bag` | `3200315` | `3200323` | `tyre_with_compost_bag_ti2v_izar_v100` |
| `cutting_board_with_rubber_duck` | `3200316` | `3200324` | `cutting_board_with_rubber_duck_ti2v_izar_v100` |

当前记录的是完整实验已提交，不能写作 3000 步训练完成。下一步检查完整作业状态、checkpoint 和视频质量。

证据：[作业与路径清单](../trained/new_demos_ti2v_runs.json) · [TI2V 运行说明](../example/new_demos_ti2v.md) · [checkpoint 检查工具](../scripts/evaluation/check_motion_smoke.py)；debug 验证结果位于各 debug 输出目录的 `debug_validation.json`。本次代码修改尚未新建 commit。


### 2026-10-06｜重组织三阶段仓库；CHORD smoke 与阶段二复现

**目的**：在 `vmmpm` 分支搭建逐 case 的三阶段入口：先 CHORD 优化主动运动，再以主动 mesh 轨迹驱动被动体 MPM，第三阶段联合优化暂留未实现。以下实验于 10 月 6 日执行；最终结果于 10 月 7 日核对，归入执行日。

#### 代码与目录重组织

- 根目录 `train.py` 保留统一命令入口，参数解析和调度移入 `pipeline/runner.py`；原 CHORD 训练移入 `pipeline/stages/chord.py`，保留原优化逻辑与参数，增加最终 iteration 的显式保存条件。
- 支持 `--stage 1`、`--stage 2`、`--stage 1 2`、`--stage 3`；case 名从 `data/` 子目录发现，例如 `--cat_with_cushion`，亦可使用 `--case cat_with_cushion`。阶段一 case 选择提供路径和可用的 prompt 文件，不自动套用历史超参数。
- `tiny_vae` 权重及 Wan2.2-I2V-A14B 软连接迁入 `ckpts/`，同步更新默认路径、脚本和文档；权重与本地资产不进入 Git。原入口用户备份保持在 `archived_chord/train.py`。
- 原 Slurm 脚本移入 `scripts/slurm/`，三阶段入口验证脚本单独放在 `scripts/slurm_sss/`；历史实验归档到 `archived_chord/scripts/`，上游说明及展示资料保留在 `archived_chord/upstream/`。
- 复用垫子 height-field 体积化工具到 `physics/geometry/`，主动猫 mesh 导出到 `scripts/evaluation/`；本地求解器移到 `physics/warp_mpm/` 并更新导入。`wan`、LightX2V 和 simple-knn 保留原导入/构建路径。更新新 README 与归档迁移索引。

#### 阶段一：五步真实 CHORD 验证

作业 **3204378**，`qos=debug`，节点 i53，COMPLETED / 0:0，耗时 6:11。使用 `chord0`，猫垫场景执行 5 步后正常退出，包含实际 SDS/反传及 checkpoint 保存。猫和垫子 checkpoint 均有限；第 1→5 步运动参数最大变化分别为 0.0228262、0.0234323。请求 416×240，Wan 实际渲染网格 416×224。

该 smoke 支持重组织后的基础训练链路可运行，不验证收敛、视频质量或第 300 步附加控制点阶段。

[提交脚本](../scripts/slurm_sss/cat_with_cushion_stage1_debug_izar.sh) · [检查结果](../trained/cat_with_cushion/cat_with_cushion_stage1_debug_3204378/debug_validation.json)。

#### 阶段二：接入现有猫轨迹 → MPM 粒子和视频

阶段二目前仅支持猫垫案例：复用历史 CHORD 3000 步 checkpoint 导出 41 帧主动猫 mesh，再复用零静置 MPM 驱动代码产生被动体粒子轨迹和视频。**没有使用五步 smoke 运动作为物理输入，没有实现粒子到被动体 4D mesh，也没有实现第三阶段联合优化。** 仿真主程序为 `physics/cat_cushion_mpm.py`，最终固定使用 `chord0` 和本地 `physics/warp_mpm`。

故障与对照记录：

- **3204392**：旧 `mpmavatar` 环境 Python 可执行文件缺失，仿真前失败。后续该环境恢复后试跑 **3204728**，仍因旧 Warp 不支持本地代码的 `wp.matrix_from_rows` 而 FAILED / 1:0；没有修改环境来作为最终验证依据。
- **3204399 / 3204403 / 3204407**：导出初始化耗时较长，诊断定位到创建首个 CUDA 张量；任务取消后保留日志。未证明节点或环境是唯一原因，不能据这些取消作业宣称 MPM 不可运行。
- **3204410**：直接 E1 smoke 在 i39 未完成并取消；相同脚本换节点后的 **3204426** 在 i10、`chord0` 下 COMPLETED / 0:0，耗时 2:37、结果有限。E1 使用规定下压/撤回轨迹，不是 CHORD 轨迹驱动，作业完成也不能单独证明求解器与历史 E2 数值等价。
- **3204729**：主动猫导出和 MPM 初始化已完成，但新增源码哈希记录对 Warp struct 调用 `inspect.getfile()` 导致 TypeError。修复为使用定义模块的 `__file__`；这是本次重组织引入的记录逻辑错误。
- **3204979**：该错误修复后进入真实仿真，但 0.033333 s 即提前停止，3499/4763 粒子触边、速度最大约 1321.46；数值有限但 completed=false。仅有失败前短视频，作业 FAILED / 1:0，耗时 3:48。

#### 根因定位与最终复现

用 `git show dev_mpm:...` 对照旧代码，无需切分支。确认主动猫导出只有迁移根目录定位变化；驱动 kernel 与视频 renderer 保持一致。当前主动猫顶点、面、FPS、center、scale 与历史 E2 导出逐项完全相同，最大差为 0；失败运行的物理配置与历史成功零静置实验一致，所读外部参考求解器源码哈希也匹配历史记录。

关键差异为旧 Warp 的 `wp.mat33(vec, vec, vec)` **按列构造**，本地兼容代码却替换为 `wp.matrix_from_rows`，而权重索引仍为 `w[0,i] * w[1,j] * w[2,k]`。独立样例的 27 个插值权重总和从应有的 1 变成 0.48759975。将本地对应构造改为 **`wp.matrix_from_cols`**，恢复历史语义，保持材料、边界、时间步、轨迹不变。另发现本地 separate/slip 缺少历史速度投影修复，但本轮用 sticky，未将其混入对照修改。

最终作业 **3205063**，`qos=debug`，i04，COMPLETED / 0:0，耗时 **7:30**；文件显示 10 月 6 日晚生成，10 月 7 日核对：

- 输出 **55 帧 × 4763 粒子**，所有位置有限、completed=true、触边粒子峰值 0；MP4 全部 55 帧可解码。
- 局部平均峰值下压 **0.0170468912**，历史零静置实验 3198923 为 **0.0170468614**；末帧约 0.00953117。
- 与历史 3198923 完整粒子轨迹逐点对照，最大位置差 **1.62125e-5**（归一化坐标）。

结论：支持矩阵行列构造错误是本次异常响应的原因；恢复列构造后，猫垫阶段二粒子/视频管线成功运行，基本复现历史零静置结果。不是严格物理正确、无穿透或接触收敛的证明；未得到被动体 4D mesh，未接入联合优化。下一步再明确粒子/表面/CHORD 表征接口。本轮未新建 Git commit。

[阶段二提交脚本](../scripts/slurm_sss/cat_with_cushion_stage2_debug_izar.sh) · [MPM 视频](../trained/cat_with_cushion/stage2_debug_3205063/mpm/mpm_process.mp4) · [结果](../trained/cat_with_cushion/stage2_debug_3205063/mpm/summary.json) · [配置](../trained/cat_with_cushion/stage2_debug_3205063/mpm/config.json)。

来源：[当天证据索引](experiment_diary_sources.md#d20261006)。
