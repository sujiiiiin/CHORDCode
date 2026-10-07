# 归档与迁移索引

历史脚本从仓库根目录运行。本次只更新路径/导入，保留原算法和局限。

| 原位置 | 新位置 |
|---|---|
| 原 train.py | archived_chord/train.py（用户备份，未改动） |
| 原 README.md、assets/ | archived_chord/upstream/ |
| example/ 普通脚本 | archived_chord/upstream/example/ |
| example/*_izar_v100.sh | scripts/slurm/ |
| scripts/experiments/ | archived_chord/scripts/experiments/ |
| scripts/experiments_mpm/ | archived_chord/scripts/experiments_mpm/ |
| E1 cushion_volume_particles.py | physics/geometry/cushion_volume_particles.py |
| E2 export_chord_cat_motion.py | scripts/evaluation/export_chord_cat_motion.py |
| example/new_demos_ti2v.md | docs/new_demos_ti2v.md |
| tiny_vae/ 和 Wan 权重软连接 | ckpts/ |

3Dassets/ 为用户事先迁入的资产目录，已忽略 Git；资产预处理工具已更新引用。
历史 E3 仍依赖仓库外 MPMAvatar，不是统一阶段二。
研究日记与原始备份保留历史路径作为证据，现位置可通过此表查找。

本地 `warp_mpm/` 已迁入 `physics/warp_mpm/`，历史 E1 的导入同步更新。
