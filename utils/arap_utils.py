import numpy as np
import pytorch3d
import pytorch3d.ops
import torch

from gaussian_renderer.dynamic_renderer import render_dynamic_whole

svd = torch.svd


def geodesic_distance_floyd(cur_node, K=8):
    node_num = cur_node.shape[0]
    nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(
        cur_node[None], cur_node[None], None, None, K=K + 1
    )
    nn_dist, nn_idx = nn_dist[0] ** 0.5, nn_idx[0]
    dist_mat = torch.inf * torch.ones(
        [node_num, node_num], dtype=torch.float32, device=cur_node.device
    )
    dist_mat.scatter_(dim=1, index=nn_idx, src=nn_dist)
    dist_mat = torch.minimum(dist_mat, dist_mat.T)
    for i in range(nn_dist.shape[0]):
        dist_mat = torch.minimum(dist_mat[:, i, None] + dist_mat[None, i, :], dist_mat)
    return dist_mat


def cal_connectivity_from_points(
    points=None,
    radius=0.1,
    K=10,
    trajectory=None,
    least_edge_num=3,
    node_radius=None,
    mode="nn",
    GraphK=4,
    adaptive_weighting=True,
):
    Nv = points.shape[0] if points is not None else trajectory.shape[0]
    if trajectory is None:
        if mode == "floyd":
            dist_mat = geodesic_distance_floyd(points, K=GraphK) ** 2
            mask = torch.eye(Nv, device=dist_mat.device, dtype=torch.bool)
            dist_mat[mask] = torch.inf
            nn_dist, nn_idx = dist_mat.sort(dim=1)
            nn_dist, nn_idx = nn_dist[:, :K], nn_idx[:, :K]
        else:
            knn_res = pytorch3d.ops.knn_points(points[None], points[None], None, None, K=K + 1)
            nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]
    else:
        trajectory = trajectory.reshape([Nv, -1]) / trajectory.shape[1]
        if mode == "floyd":
            dist_mat = geodesic_distance_floyd(trajectory, K=GraphK) ** 2
            mask = torch.eye(Nv, device=dist_mat.device, dtype=torch.bool)
            dist_mat[mask] = torch.inf
            nn_dist, nn_idx = dist_mat.sort(dim=1)
            nn_dist, nn_idx = nn_dist[:, :K], nn_idx[:, :K]
        else:
            knn_res = pytorch3d.ops.knn_points(
                trajectory[None], trajectory[None], None, None, K=K + 1
            )
            nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]

    nn_idx[:, least_edge_num:] = torch.where(
        nn_dist[:, least_edge_num:] < radius**2,
        nn_idx[:, least_edge_num:],
        -torch.ones_like(nn_idx[:, least_edge_num:]),
    )
    nn_dist[:, least_edge_num:] = torch.where(
        nn_dist[:, least_edge_num:] < radius**2,
        nn_dist[:, least_edge_num:],
        torch.ones_like(nn_dist[:, least_edge_num:]) * torch.inf,
    )

    if adaptive_weighting:
        weight = torch.exp(-nn_dist / nn_dist.mean())
    elif node_radius is None:
        weight = torch.exp(-nn_dist)
    else:
        nn_radius = node_radius[nn_idx]
        weight = torch.exp(-nn_dist / (2 * nn_radius**2))
    weight = weight / weight.sum(dim=-1, keepdim=True)

    device = nn_idx.device
    ii = torch.arange(Nv, device=device)[:, None].long().expand(Nv, K).reshape([-1])
    jj = nn_idx.reshape([-1])
    nn = torch.arange(K, device=device)[None].long().expand(Nv, K).reshape([-1])
    mask = jj != -1
    return ii[mask], jj[mask], nn[mask], weight


def produce_edge_matrix_nfmt(verts: torch.Tensor, edge_shape, ii, jj, nn, device="cuda"):
    edges = torch.zeros(edge_shape, device=device)
    edges[ii, nn] = verts[ii] - verts[jj]
    return edges


def estimate_rotation(source, target, ii, jj, nn, K=10, weight=None, sample_idx=None):
    Nv = len(source)
    source_edge_mat = produce_edge_matrix_nfmt(source, (Nv, K, 3), ii, jj, nn)
    target_edge_mat = produce_edge_matrix_nfmt(target, (Nv, K, 3), ii, jj, nn)
    if weight is None:
        weight = torch.zeros(Nv, K, device=source.device)
        weight[ii, nn] = 1
    if sample_idx is not None:
        source_edge_mat = source_edge_mat[sample_idx]
        target_edge_mat = target_edge_mat[sample_idx]
    D = torch.diag_embed(weight, dim1=1, dim2=2)
    S = torch.bmm(source_edge_mat.permute(0, 2, 1), torch.bmm(D, target_edge_mat))
    unchanged_verts = torch.unique(
        torch.where((source_edge_mat == target_edge_mat).all(dim=1))[0]
    )
    S[unchanged_verts] = 0

    U, sig, W = svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))
    entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()
    if len(entries_to_flip) > 0:
        Umod = U.clone()
        cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)
        Umod[entries_to_flip, :, cols_to_flip] *= -1
        R[entries_to_flip] = torch.bmm(
            W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1)
        )
    return R


def cal_arap_error(
    nodes_sequence,
    ii,
    jj,
    nn,
    K=10,
    weight=None,
    sample_num=512,
    sampled_idx=None,
    sample_times=1,
):
    Nt, Nv, _ = nodes_sequence.shape
    arap_error = 0
    if weight is None:
        weight = torch.zeros(Nv, K, device=nodes_sequence.device)
        weight[ii, nn] = 1

    source_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[0], (Nv, K, 3), ii, jj, nn)
    if Nv > sample_num:
        if sampled_idx is not None:
            sample_idx = sampled_idx
        else:
            sample_idx = torch.cat(
                [
                    torch.from_numpy(np.random.choice(Nv, sample_num))
                    .long()
                    .to(nodes_sequence.device)
                    for _ in range(sample_times)
                ],
                dim=0,
            )
    else:
        sample_idx = torch.cat(
            [torch.arange(Nv, device=nodes_sequence.device) for _ in range(sample_times)],
            dim=0,
        )

    sampled_source_edge = source_edge_mat[sample_idx]
    sampled_weight = weight[sample_idx]
    for idx in range(1, Nt):
        with torch.no_grad():
            rotation = estimate_rotation(
                nodes_sequence[0],
                nodes_sequence[idx],
                ii,
                jj,
                nn,
                K=K,
                weight=sampled_weight,
                sample_idx=sample_idx,
            )
        target_edge_mat = produce_edge_matrix_nfmt(
            nodes_sequence[idx], (Nv, K, 3), ii, jj, nn
        )[sample_idx]
        rot_rigid = torch.bmm(rotation, sampled_source_edge.permute(0, 2, 1)).permute(0, 2, 1)
        stretch_norm = torch.norm(target_edge_mat - rot_rigid, dim=2) ** 2
        arap_error += (sampled_weight * stretch_norm).sum()
    return arap_error


def landmark_interpolate(landmarks, steps, step, interpolation="log"):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    if stage == 0:
        return 0

    ldm1, ldm2 = landmarks[stage - 1], landmarks[stage]
    if ldm2 <= 0:
        return 0
    step1, step2 = steps[stage - 1], steps[stage]
    ratio = (step - step1) / (step2 - step1)
    if interpolation == "log":
        return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
    if interpolation == "linear":
        return ldm1 * (1 - ratio) + ldm2 * ratio
    raise NotImplementedError(f"Unknown interpolation type: {interpolation}")


def calc_temp_loss_obj(gaussians, max_time: int, view_mats, Ks, image_width, image_height):
    means3D, rotations = gaussians.get_xyz_rotation_whole()
    cur_xyz = means3D[1:max_time]
    prev_xyz = means3D[: max_time - 1]
    displacement_tensor = cur_xyz - prev_xyz
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    view_mats = view_mats.unsqueeze(0).repeat(max_time - 1, 1, 1, 1)
    Ks = Ks.unsqueeze(0).repeat(max_time - 1, 1, 1, 1)
    displacement_map = render_dynamic_whole(
        view_mats,
        Ks,
        gaussians,
        image_width,
        image_height,
        None,
        max_time - 1,
        bg_color,
        override_color=displacement_tensor,
        detach=True,
        mem_means=means3D[: max_time - 1],
        mem_rotations=rotations[: max_time - 1],
    )
    return displacement_map.square().sum()


def calc_temp_loss(dynamic_scene, max_time: int, view_mats, Ks, image_width, image_height):
    loss = 0.0
    for gaussians, is_static in dynamic_scene.dynamic_gaussians.values():
        if is_static:
            continue
        loss += calc_temp_loss_obj(
            gaussians, max_time, view_mats, Ks, image_width, image_height
        )
    return loss


def arap_loss(nodes_t, sample_num=1024, sample_times=1, stored_coninfo=None):
    nodes_t = nodes_t.permute(1, 0, 2)
    hyper_nodes = nodes_t[:, 0]
    if stored_coninfo is None:
        ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, K=10)
    else:
        ii = stored_coninfo["ii"]
        jj = stored_coninfo["jj"]
        nn = stored_coninfo["nn"]
        weight = stored_coninfo["weight"]
    error = cal_arap_error(
        nodes_t.permute(1, 0, 2),
        ii,
        jj,
        nn,
        sample_num=sample_num,
        weight=None,
        sample_times=sample_times,
    )
    return error, {"ii": ii, "jj": jj, "nn": nn, "weight": weight}
