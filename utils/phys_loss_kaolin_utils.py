import torch
from pytorch3d.ops import knn_points


def inside_barriers(query_pts, barriers):
    if not isinstance(query_pts, torch.Tensor):
        query_pts = torch.tensor(query_pts, dtype=torch.float32).cuda()
    while len(query_pts.shape) < 3:
        query_pts = query_pts.unsqueeze(0)

    for in_barrier, out_barrier in barriers.values():
        in_barrier = in_barrier.unsqueeze(0)
        out_barrier = out_barrier.unsqueeze(0)
        with torch.no_grad():
            _, _, nn_in = knn_points(query_pts, in_barrier, K=1, return_nn=True)
            _, _, nn_out = knn_points(query_pts, out_barrier, K=1, return_nn=True)
        dist_in = ((nn_in[0] - query_pts).square().sum(dim=-1) ** 0.5).squeeze(-1)
        dist_out = ((nn_out[0] - query_pts).square().sum(dim=-1) ** 0.5).squeeze(-1)
        if (dist_in < dist_out).sum() != 0:
            return True
    return False
