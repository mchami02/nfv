import numpy as np
import torch


def _lax_hopf_core(ic, xmax, x_positions, nt, dt, flow, dtype, device):
    """Core Lax-Hopf computation at arbitrary x positions.

    Args:
        ic: list of PiecewiseConstant ICs
        xmax: domain length
        x_positions: 1D tensor of shape (Nx,) — where to evaluate solution
        nt: number of time steps
        dt: time step
        flow: flow model
        dtype, device: torch config

    Returns:
        rho: tensor of shape (batch, nt, Nx)
    """
    if dtype == torch.float32:
        raise ValueError(
            "Lax hopf solver requires torch64 dtype to be accurate (for 32 bit to work, remove the lines ic_xs[:, 0] = -1e9 and ic_xs[:, -1] = 1e9)"
        )

    Nx = len(x_positions)
    Nt = nt
    tmax = dt * Nt

    ic_ks = np.array([x.ks for x in ic])
    ic_ks = torch.from_numpy(ic_ks).to(dtype).to(device)

    flow = {"vmax": flow.vmax, "w": flow.w, "q": flow.q, "qp": flow.qp, "R": flow.R, "Rp": flow.Rp, "kmax": flow.kmax}

    batch_size = ic_ks.shape[0]

    ic_xs = torch.Tensor(np.array([ic_i.xs * xmax for ic_i in ic])).to(device).to(dtype)

    # consider it as a problem on R
    ic_xs[:, 0] = -1e9
    ic_xs[:, -1] = 1e9

    ic_ks = ic_ks.to(device)

    # precompute bi's
    bi = []
    for i in range(ic_xs.shape[1] - 1):
        b = ic_ks[:, i] * ic_xs[:, i]
        for l in range(i):
            b -= (ic_xs[:, l + 1] - ic_xs[:, l]) * ic_ks[:, l]
        bi.append(b)
    bi = torch.stack(bi).to(device).T

    t = torch.linspace(0, tmax - dt, Nt, device=device, dtype=dtype).view(1, -1, 1, 1)
    t[:, 0, :, :] = 1e-9
    x = x_positions.view(1, 1, -1, 1)

    xi = ic_xs.view(batch_size, 1, 1, -1)

    conditionJ = xi > x - flow["vmax"] * t
    conditionU = xi < x - flow["w"] * t

    Jl = torch.where(conditionJ.any(dim=-1), torch.clamp(torch.argmax(conditionJ.to(dtype), dim=-1) - 1, min=0), len(xi) - 2) * torch.ones_like(
        conditionJ.any(dim=-1)
    )

    Ju = torch.where(
        conditionU.any(dim=-1),
        torch.clamp(xi.shape[-1] - torch.argmax(conditionU.int().flip(-1), dim=-1) - 1, max=xi.shape[-1] - 2),
        torch.zeros_like(conditionU.any(dim=-1)),
    )
    del conditionJ, conditionU

    xi = xi.expand(-1, t.shape[1], x.shape[2], -1)
    ki = ic_ks.view(batch_size, 1, 1, -1).expand(-1, t.shape[1], x.shape[2], -1)
    bi = bi.view(batch_size, 1, 1, -1).expand(-1, t.shape[1], x.shape[2], -1)

    max_range_len = (Ju - Jl + 1).max().item()
    i_range = Ju.unsqueeze(-1).expand(batch_size, t.shape[1], x.shape[2], max_range_len).clone()
    range_tensor = torch.arange(max_range_len, device=device).view(1, 1, 1, -1)
    i_range = torch.minimum(Jl.unsqueeze(-1) + range_tensor, Ju.unsqueeze(-1))
    del range_tensor, Ju, Jl

    xi_range_p1 = xi.gather(3, i_range + 1)
    xi_range = xi.gather(3, i_range)
    ki_range = ki.gather(3, i_range)
    bi_range = bi.gather(3, i_range)
    M_values = _Mc0(t, x, xi_range, xi_range_p1, ki_range, bi_range, flow)
    i_store = i_range.gather(3, torch.argmin(M_values, dim=-1, keepdim=True))
    del xi_range_p1, xi_range, ki_range, bi_range, M_values, i_range

    xip1 = xi.gather(3, i_store + 1)
    xi = xi.gather(3, i_store)
    ki = ki.gather(3, i_store)
    del i_store
    rho = _rho_c0(t.squeeze(-1), x.squeeze(-1), xi.squeeze(-1), xip1.squeeze(-1), ki.squeeze(-1), flow=flow)

    return rho


def LaxHopf(ic, nx, nt, dx, dt, flow, dtype, device, **kwargs):
    n_points_integral = 10
    Nx = nx * n_points_integral
    xmax = dx * nx
    dx_fine = dx / n_points_integral
    x_positions = torch.linspace(0, xmax - dx_fine, Nx, device=device, dtype=dtype)

    rho = _lax_hopf_core(ic, xmax, x_positions, nt, dt, flow, dtype, device)

    # Average 10 sub-cell points per cell
    rho = rho.view(-1, nt, nx, n_points_integral).mean(dim=-1)

    # Override t=0 with discretized IC (cell averages)
    ic_discretized = np.array([ic_i.discretize(nx) for ic_i in ic])
    rho[:, 0, :] = torch.from_numpy(ic_discretized).to(dtype).to(device)

    return rho


def _evaluate_pointwise_ic(ic_list, nx, dtype, device):
    """Evaluate piecewise-constant ICs at cell centers (pointwise, not averaged)."""
    centers = np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)
    result = np.empty((len(ic_list), nx))
    for b, ic in enumerate(ic_list):
        indices = np.searchsorted(ic.xs, centers, side='right') - 1
        indices = np.clip(indices, 0, len(ic.ks) - 1)
        result[b] = ic.ks[indices]
    return torch.from_numpy(result).to(dtype).to(device)


def LaxHopfPointWise(ic, nx, nt, dx, dt, flow, dtype, device, **kwargs):
    xmax = dx * nx
    x_positions = torch.linspace(dx / 2, xmax - dx / 2, nx, device=device, dtype=dtype)

    rho = _lax_hopf_core(ic, xmax, x_positions, nt, dt, flow, dtype, device)

    # Override t=0 with pointwise IC at cell centers
    rho[:, 0, :] = _evaluate_pointwise_ic(ic, nx, dtype, device)

    return rho


def _Mc0(t, x, xi, xip1, ki, bi, flow):
    c1 = xi + t * flow["w"]
    c2 = xi + t * flow["qp"](ki)
    c3 = xip1 + t * flow["qp"](ki)
    # c4 = xip1 + t * flow['vmax']

    M = torch.where(
        (x >= c1) & (x < c2),
        t * flow["R"]((x - xi) / t) - ki * xi + bi,
        torch.where(
            (x >= c2) & (x < c3),
            t * flow["q"](ki) - ki * x + bi,
            t * flow["R"]((x - xip1) / t) - ki * xip1 + bi,
            # torch.where(
            #     (x >= c3) & (x <= c4),
            #     t * flow['R']((x - xip1) / t, flow) - ki * xip1 + bi,
            #     torch.full_like(M_condition_1, float('inf')))
        ),
    )

    return M


def _rho_c0(t, x, xi, xip1, ki, flow):
    t_flow_qp_ki = t * flow["qp"](ki)
    c1 = xi + t * flow["w"]
    c2 = xi + t_flow_qp_ki
    c3 = xip1 + t_flow_qp_ki
    # c4 = xip1 + t * flow['vmax']

    return torch.where(
        (x >= c1) * (x < c2),
        -flow["Rp"]((x - xi) / t),
        torch.where(
            (x >= c2) * (x < c3),
            ki,
            -flow["Rp"]((x - xip1) / t),
            # torch.where(
            #     (x >= c3) * (x <= c4),
            #     -flow['Rp']((x - xip1)/t, flow),
            #     torch.zeros_like(x)
            # )
        ),
    )  # .squeeze(-1)
