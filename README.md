# CHORD–MPM: per-case motion optimization

本仓库基于已有 CHORD 与 MPM 实验组织三阶段 pipeline。视频模型权重固定，每个 case 独立优化运动。

| 阶段 | 目标 | 状态 |
|---|---|---|
| 1 CHORD | 视频监督下优化控制点 | 复用原训练；主动体 mesh 导出仍为独立工具 |
| 2 MPM | 主动体运动驱动被动体 | 猫垫案例可输出粒子轨迹和视频；4D mesh 未实现 |
| 3 Joint | 联合优化 | 未实现 |

## 使用

从仓库根目录，在原 CHORD 环境中运行：

```bash
python train.py --help
python train.py --stage 1 --ca_with_dog \
  --task i2v-A14B --size '416*240' \
  --ckpt_dir ./ckpts/Wan2.2-I2V-A14B \
  --prompt "The man bends down and rubs the dog's head."
```

case 名来自 `data/` 子目录；`--case ca_with_dog` 与 `--ca_with_dog` 等价。
原 CHORD 参数继续支持。case 选择设置输入/模型目录及可用的 `prompt.txt`，不会套用历史超参数。
既有完整运行配置和 Slurm 资源参数保留在 `scripts/slurm/`。

`--stage 1 2` 可顺序运行猫垫案例；阶段二需要 `--mpm-output-dir`。
`--stage 3` 仍在训练启动前明确报未实现。
见 [入口说明](docs/pipeline_entrypoint.md)。

## 目录

- `train.py`、`pipeline/`：入口、调度和三阶段模块。
- `scene/`、`arguments/`、`utils/`：现有 CHORD 表征与优化工具。
- `gaussian_renderer/`、`mesh_renderer/`：现有渲染模块。
- `wan/`：视频监督后端。`lightx2v/` 与 `lightx2v_platform/` 保留现有可选依赖和 Tiny VAE 导入路径。
- `physics/warp_mpm/`：现有本地求解器；恢复历史矩阵列构造语义后，猫垫零静置轨迹已基本复现（并非任意场景等价性证明）。
- `physics/geometry/`：复用垫子 height-field 体积化工具，不是通用体积建模。
- `scripts/preprocess/`、`scripts/evaluation/`、`scripts/prompting/`：现有工具。主动猫 mesh 导出位于 `scripts/evaluation/export_chord_cat_motion.py`。
- `scripts/slurm/`：原有单独 CHORD 提交脚本。
- `scripts/slurm_sss/`：三阶段入口的阶段一和阶段二 debug 提交脚本。
- `ckpts/`：本地权重和权重软连接，不进入 Git。
- `submodules/simple-knn/`：保留原 CUDA 扩展和 Git submodule。
- `data/`、`trained/`、`logs/`：既有输入输出，保留原地。
- `archived_chord/`：原入口备份、上游展示资料和历史实验。
- `mydocs/`：本地研究日记。

统一 case 配置、阶段产物协议、通用求解器适配器和被动体表面绑定均未实现。
阶段二复用 `physics/cat_cushion_mpm.py`，使用仓库内 `physics/warp_mpm` 求解器和 `chord0` 环境。
本次重组织没有新增物理或优化算法。

## 权重与环境

```text
ckpts/
├── tiny_vae/lighttaew2_1.pth
└── Wan2.2-I2V-A14B -> /scratch/izar/ydu/checkpoints/Wan2.2-I2V-A14B
```

Tiny VAE 默认路径为 `ckpts/tiny_vae/lighttaew2_1.pth`，可用 `--tiny_vae_path` 覆盖。
Wan 权重通过 `--ckpt_dir` 指定；已有 TI2V 外部权重路径不变。
本次未重建 CHORD 或 MPM 环境。安装细节保留在原 README，以上权重路径取代其中的历史路径。

参见 [预处理](docs/preprocess.md)、[显存设置](docs/memory_constrained_training.md)、[TI2V 运行记录](docs/new_demos_ti2v.md)。

## 来源

本项目基于 CHORD、Wan、MPMAvatar 及相关依赖。原始代码版权头保留，上游介绍、引用与致谢见
[原 README](archived_chord/upstream/README.md)。上游适用的许可条款仍然有效。
