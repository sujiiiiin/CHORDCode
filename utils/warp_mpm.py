"""A minimal 3D MLS-MPM solver used only for direct mechanism experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.func
def _weight(f: float, offset: int):
    result = 0.0
    if offset == 0:
        result = 0.5 * (1.5 - f) * (1.5 - f)
    elif offset == 1:
        result = 0.75 - (f - 1.0) * (f - 1.0)
    else:
        result = 0.5 * (f - 0.5) * (f - 0.5)
    return result


@wp.func
def _grid_index(ix: int, iy: int, iz: int, ny: int, nz: int):
    return ix * ny * nz + iy * nz + iz


@wp.kernel
def _clear_grid(grid_v: wp.array(dtype=wp.vec3), grid_m: wp.array(dtype=float)):
    index = wp.tid()
    grid_v[index] = wp.vec3(0.0, 0.0, 0.0)
    grid_m[index] = 0.0


@wp.kernel
def _p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    f: wp.array(dtype=wp.mat33),
    mass: float,
    volume: float,
    mu: float,
    lam: float,
    dt: float,
    dx: float,
    inv_dx: float,
    origin: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
):
    p = wp.tid()
    xp = (x[p] - origin) * inv_dx
    base = wp.vec3i(int(xp[0] - 0.5), int(xp[1] - 0.5), int(xp[2] - 0.5))
    fx = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))

    deformation = f[p]
    identity = wp.identity(n=3, dtype=float)
    j = wp.max(wp.determinant(deformation), 0.05)
    kirchhoff = mu * (deformation * wp.transpose(deformation) - identity) + lam * wp.log(j) * identity
    stress = (-dt * volume * 4.0 * inv_dx * inv_dx) * kirchhoff
    affine = stress + mass * c[p]

    for i in range(3):
        for j_offset in range(3):
            for k in range(3):
                ix = base[0] + i
                iy = base[1] + j_offset
                iz = base[2] + k
                if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                    weight = _weight(fx[0], i) * _weight(fx[1], j_offset) * _weight(fx[2], k)
                    dpos = (wp.vec3(float(i), float(j_offset), float(k)) - fx) * dx
                    momentum = mass * v[p] + affine * dpos
                    index = _grid_index(ix, iy, iz, ny, nz)
                    wp.atomic_add(grid_v, index, weight * momentum)
                    wp.atomic_add(grid_m, index, weight * mass)


@wp.func
def _contact_velocity(
    position: wp.vec3,
    velocity: wp.vec3,
    center: wp.vec3,
    collider_velocity: wp.vec3,
    radius: float,
    friction: float,
):
    delta = position - center
    distance = wp.length(delta)
    result = velocity
    if distance < radius and distance > 1.0e-8:
        normal = delta / distance
        relative = velocity - collider_velocity
        normal_velocity = wp.dot(relative, normal)
        if normal_velocity < 0.0:
            tangent = relative - normal_velocity * normal
            tangent_length = wp.length(tangent)
            if tangent_length > 1.0e-8:
                scale = wp.max(0.0, 1.0 - friction * wp.abs(normal_velocity) / tangent_length)
                tangent = tangent * scale
            relative = tangent
            result = collider_velocity + relative
    return result


@wp.kernel
def _grid_update(
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3,
    dx: float,
    origin: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
    collider_center: wp.vec3,
    collider_velocity: wp.vec3,
    collider_radius: float,
    friction: float,
    collider_enabled: int,
):
    index = wp.tid()
    node_mass = grid_m[index]
    if node_mass > 0.0:
        ix = index // (ny * nz)
        remainder = index - ix * ny * nz
        iy = remainder // nz
        iz = remainder - iy * nz
        velocity = grid_v[index] / node_mass + dt * gravity
        if ix < 2 and velocity[0] < 0.0:
            velocity = wp.vec3(0.0, velocity[1], velocity[2])
        if ix > nx - 3 and velocity[0] > 0.0:
            velocity = wp.vec3(0.0, velocity[1], velocity[2])
        if iy < 2 and velocity[1] < 0.0:
            velocity = wp.vec3(velocity[0], 0.0, velocity[2])
        if iy > ny - 3 and velocity[1] > 0.0:
            velocity = wp.vec3(velocity[0], 0.0, velocity[2])
        if iz < 2 and velocity[2] < 0.0:
            velocity = wp.vec3(velocity[0], velocity[1], 0.0)
        if iz > nz - 3 and velocity[2] > 0.0:
            velocity = wp.vec3(velocity[0], velocity[1], 0.0)
        if collider_enabled == 1:
            position = origin + wp.vec3(float(ix), float(iy), float(iz)) * dx
            velocity = _contact_velocity(
                position,
                velocity,
                collider_center,
                collider_velocity,
                collider_radius,
                friction,
            )
        grid_v[index] = velocity


@wp.kernel
def _g2p(
    x: wp.array(dtype=wp.vec3),
    rest_x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    f: wp.array(dtype=wp.mat33),
    fixed: wp.array(dtype=wp.int32),
    dt: float,
    dx: float,
    inv_dx: float,
    origin: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
    damping: float,
    collider_center: wp.vec3,
    collider_velocity: wp.vec3,
    collider_radius: float,
    friction: float,
    collider_enabled: int,
    particle_projection_enabled: int,
    grid_v: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    xp = (x[p] - origin) * inv_dx
    base = wp.vec3i(int(xp[0] - 0.5), int(xp[1] - 0.5), int(xp[2] - 0.5))
    fx = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_c = wp.mat33(0.0)
    for i in range(3):
        for j_offset in range(3):
            for k in range(3):
                ix = base[0] + i
                iy = base[1] + j_offset
                iz = base[2] + k
                if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                    weight = _weight(fx[0], i) * _weight(fx[1], j_offset) * _weight(fx[2], k)
                    dpos = (wp.vec3(float(i), float(j_offset), float(k)) - fx) * dx
                    node_v = grid_v[_grid_index(ix, iy, iz, ny, nz)]
                    new_v = new_v + weight * node_v
                    new_c = new_c + 4.0 * inv_dx * inv_dx * weight * wp.outer(node_v, dpos)
    new_v = new_v * damping
    new_x = x[p] + dt * new_v
    if collider_enabled == 1 and particle_projection_enabled == 1:
        delta = new_x - collider_center
        distance = wp.length(delta)
        if distance < collider_radius and distance > 1.0e-8:
            normal = delta / distance
            new_x = collider_center + normal * collider_radius
            new_v = _contact_velocity(
                new_x,
                new_v,
                collider_center,
                collider_velocity,
                collider_radius,
                friction,
            )
    if fixed[p] == 1:
        x[p] = rest_x[p]
        v[p] = wp.vec3(0.0, 0.0, 0.0)
        c[p] = wp.mat33(0.0)
        f[p] = wp.identity(n=3, dtype=float)
    else:
        x[p] = new_x
        v[p] = new_v
        c[p] = new_c
        f[p] = (wp.identity(n=3, dtype=float) + dt * new_c) * f[p]


@dataclass(frozen=True)
class MPMConfig:
    dx: float
    dt: float
    density: float
    particle_volume: float
    mu: float
    lam: float
    damping: float = 0.9995
    gravity_y: float = 0.0
    padding_cells: int = 5


class WarpMPMSolver:
    def __init__(
        self,
        points: np.ndarray,
        fixed: np.ndarray,
        config: MPMConfig,
        device: str,
        initial_positions: np.ndarray | None = None,
        initial_deformation: np.ndarray | None = None,
    ):
        wp.init()
        self.device = device
        self.config = config
        points = np.asarray(points, dtype=np.float32)
        fixed = np.asarray(fixed, dtype=np.int32)
        initial_positions = points if initial_positions is None else np.asarray(initial_positions, dtype=np.float32)
        if initial_positions.shape != points.shape:
            raise ValueError("initial_positions must have the same shape as points")
        identity = np.tile(np.eye(3, dtype=np.float32), (len(points), 1, 1))
        initial_deformation = (
            identity
            if initial_deformation is None
            else np.asarray(initial_deformation, dtype=np.float32)
        )
        if initial_deformation.shape != identity.shape:
            raise ValueError("initial_deformation must have shape [N, 3, 3]")
        padding = config.padding_cells * config.dx
        domain_points = np.concatenate([points, initial_positions], axis=0)
        self.origin_np = domain_points.min(axis=0) - padding
        domain_max = domain_points.max(axis=0) + padding + np.array([0.2, 0.35, 0.2], dtype=np.float32)
        dims = np.ceil((domain_max - self.origin_np) / config.dx).astype(np.int32) + 1
        self.nx, self.ny, self.nz = (int(value) for value in dims)
        self.grid_size = self.nx * self.ny * self.nz
        self.particle_count = len(points)
        self.mass = config.density * config.particle_volume
        zeros = np.zeros((len(points), 3), dtype=np.float32)
        zero_mats = np.zeros((len(points), 3, 3), dtype=np.float32)
        self.x = wp.array(initial_positions, dtype=wp.vec3, device=device)
        self.rest_x = wp.array(points, dtype=wp.vec3, device=device)
        self.v = wp.array(zeros, dtype=wp.vec3, device=device)
        self.c = wp.array(zero_mats, dtype=wp.mat33, device=device)
        self.f = wp.array(initial_deformation, dtype=wp.mat33, device=device)
        self.fixed = wp.array(fixed, dtype=wp.int32, device=device)
        self.grid_v = wp.zeros(self.grid_size, dtype=wp.vec3, device=device)
        self.grid_m = wp.zeros(self.grid_size, dtype=float, device=device)

    def step(
        self,
        collider_center: np.ndarray,
        collider_velocity: np.ndarray,
        collider_radius: float,
        friction: float,
        collider_enabled: bool,
        particle_projection_enabled: bool = True,
    ) -> None:
        cfg = self.config
        origin = wp.vec3(*self.origin_np.tolist())
        center = wp.vec3(*np.asarray(collider_center, dtype=np.float32).tolist())
        velocity = wp.vec3(*np.asarray(collider_velocity, dtype=np.float32).tolist())
        wp.launch(_clear_grid, self.grid_size, [self.grid_v, self.grid_m], device=self.device)
        wp.launch(
            _p2g,
            self.particle_count,
            [
                self.x, self.v, self.c, self.f, self.mass, cfg.particle_volume,
                cfg.mu, cfg.lam, cfg.dt, cfg.dx, 1.0 / cfg.dx, origin,
                self.nx, self.ny, self.nz, self.grid_v, self.grid_m,
            ],
            device=self.device,
        )
        wp.launch(
            _grid_update,
            self.grid_size,
            [
                self.grid_v, self.grid_m, cfg.dt, wp.vec3(0.0, cfg.gravity_y, 0.0),
                cfg.dx, origin, self.nx, self.ny, self.nz, center, velocity,
                collider_radius, friction, int(collider_enabled),
            ],
            device=self.device,
        )
        wp.launch(
            _g2p,
            self.particle_count,
            [
                self.x, self.rest_x, self.v, self.c, self.f, self.fixed,
                cfg.dt, cfg.dx, 1.0 / cfg.dx, origin, self.nx, self.ny, self.nz,
                cfg.damping, center, velocity, collider_radius, friction,
                int(collider_enabled), int(particle_projection_enabled), self.grid_v,
            ],
            device=self.device,
        )

    def positions(self) -> np.ndarray:
        return self.x.numpy()

    def velocities(self) -> np.ndarray:
        return self.v.numpy()

    def deformation_gradients(self) -> np.ndarray:
        return self.f.numpy()
