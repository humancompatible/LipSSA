import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.patches as patches


# PLOTTER FUNCTIONS --------------------------------------------------
def get_xs_gs_maximum(net, radius, dimension, px, v=True, c_vector=None):
    diameter = radius * 2
    xs = np.zeros((px * px, dimension))
    gs = np.zeros(px * px)
    mx = -1
    j = 0
    mxx = 0
    mxy = 0
    for i in np.arange(-radius, radius, diameter / px):
        for k in np.arange(-radius, radius, diameter / px):
            x = torch.tensor([[i, k]], requires_grad=True, dtype=torch.float)
            if c_vector is None:
                g = torch.autograd.functional.jacobian(lambda x: net(x), x).norm(p=1)
            else:
                g = torch.autograd.functional.jacobian(lambda x: net(x).mv(c_vector).sum(), x).norm(p=1)

            x = x.cpu().detach().numpy()[0]
            xs[j] = x
            gs[j] = g
            j += 1
            if g > mx:
                mx = g
                mxx = i
                mxy = k
    if v:
        print(f"maximum = {mx} on ({mxx} {mxy})")
    return xs, gs, mx, mxx, mxy


def plot_grads(net, radius, xs, gs, mxx, mxy):
    grid_x, grid_y = np.mgrid[
                     xs[:, 0].min():xs[:, 0].max():100j,
                     xs[:, 1].min():xs[:, 1].max():100j
                     ]
    grid_fx = griddata(points=xs, values=gs, xi=(grid_x, grid_y), method='cubic')
    plt.figure(figsize=(6, 5))
    plt.imshow(grid_fx.T,
               extent=(xs[:, 0].min(), xs[:, 0].max(), xs[:, 1].min(), xs[:, 1].max()),
               origin='lower',
               cmap='viridis',
               aspect='auto')
    plt.colorbar(label='grad')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-radius, radius))
    plt.ylim((-radius, radius))
    plt.title(f"Heatmap of gradients over input space\n"
              f"ReLU Network {net.layer_sizes}")
    plt.tight_layout()
    plt.plot(mxx, mxy, 'ro', markersize=8, markeredgecolor='black')
    plt.show()


def plot_fixed_partitions_grads(net, radius, regs, xs, gs, mxx, mxy, description):
    grid_x, grid_y = np.mgrid[
                     xs[:, 0].min():xs[:, 0].max():100j,
                     xs[:, 1].min():xs[:, 1].max():100j
                     ]
    grid_fx = griddata(points=xs, values=gs, xi=(grid_x, grid_y), method='cubic')
    plt.figure(figsize=(6, 5))
    plt.imshow(grid_fx.T,
               extent=(xs[:, 0].min(), xs[:, 0].max(), xs[:, 1].min(), xs[:, 1].max()),
               origin='lower',
               cmap='viridis',
               aspect='auto')
    plt.colorbar(label='grad')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-radius, radius))
    plt.ylim((-radius, radius))
    plt.title(f"Subregions of input space ({description})\n"
              f"ReLU Network {net.layer_sizes}")

    ax = plt.gca()
    for r in regs:
        x1, y1 = r.lb
        x2, y2 = r.ub
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=1.5,
                                 edgecolor='white',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.plot(mxx, mxy, 'ro', markersize=8, markeredgecolor='black', label='Max f(x)')
    plt.show()


def plot_value_and_grads(net, domain, radius, dimension, is_trained, px):
    xs, gs, mx, mxx, mxy = get_xs_gs_maximum(net, radius, dimension, px=px)
    pt = domain.random_point(10000)
    vals = net(pt).detach().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sc1 = ax[0].scatter(pt[:, 0], pt[:, 1], c=vals, cmap='seismic', s=5)
    fig.colorbar(sc1, ax=ax[0], label="Value")
    ax[0].set_title("Network evaluation")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid(True)
    grid_x, grid_y = np.mgrid[
                     xs[:, 0].min():xs[:, 0].max():100j,
                     xs[:, 1].min():xs[:, 1].max():100j]
    grid_fx = griddata(points=xs, values=gs, xi=(grid_x, grid_y), method='cubic')
    im = ax[1].imshow(grid_fx.T,
                      extent=(xs[:, 0].min(), xs[:, 0].max(), xs[:, 1].min(), xs[:, 1].max()),
                      origin='lower',
                      cmap='viridis',
                      aspect='auto')
    fig.colorbar(im, ax=ax[1], label="Grad")
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_xlim((-radius, radius))
    ax[1].set_ylim((-radius, radius))
    ax[1].set_title('Heatmap of gradients over input space')
    ax[1].grid(False)
    ax[1].plot(mxx, mxy, 'ro', markersize=8, markeredgecolor='black')

    if not is_trained:
        fig.suptitle(f"ReLU Net {net.layer_sizes} before training")
    else:
        fig.suptitle(f"Trained ReLU Net {net.layer_sizes}")
    plt.show()
    print(f"maximum = {mx} on ({mxx} {mxy})")


def plot_evaluated_points(net, radius, xs, gs, mxx, mxy, regs, evaluated_points, sample_budget=None):
    grid_x, grid_y = np.mgrid[
                     xs[:, 0].min():xs[:, 0].max():100j,
                     xs[:, 1].min():xs[:, 1].max():100j
                     ]
    grid_fx = griddata(points=xs, values=gs, xi=(grid_x, grid_y), method='cubic')
    plt.figure(figsize=(6, 5))
    plt.imshow(grid_fx.T,
               extent=(xs[:, 0].min(), xs[:, 0].max(), xs[:, 1].min(), xs[:, 1].max()),
               origin='lower',
               cmap='viridis',
               aspect='auto')
    plt.colorbar(label='grad')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-radius, radius))
    plt.ylim((-radius, radius))
    if sample_budget is None:
        plt.title(f"Evaluated points\n"
                  f"ReLU Network {net.layer_sizes}")
    else:
        plt.title(f"Sample budget={sample_budget}\n"
                  f"{len(regs)} subregions ({sample_budget // len(regs)} samples per subregion)\n"
                  f"Evaluated points\n"
                  f"ReLU Network {net.layer_sizes}")
    plt.tight_layout()
    plt.plot(mxx, mxy, 'ro', markersize=8, markeredgecolor='black')

    ax = plt.gca()
    for r in regs:
        x1, y1 = r.lb
        x2, y2 = r.ub
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=1.5,
                                 edgecolor='white',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.scatter(evaluated_points[:, 0], evaluated_points[:, 1], c='k', s=5)

    plt.show()


def get_random_dataset(num_points, radius, num_lakes, dim):
    X = np.random.uniform(-radius, radius, size=(num_points, dim))
    X = torch.Tensor(X)

    def generate_random_lakes(num_lakes, r, dim):
        lks = []
        for _ in range(num_lakes):
            coords = tuple(np.random.uniform(-r, r, size=dim))
            size = np.random.uniform(0.1 * r, 0.4 * r)
            lks.append((coords, size))
        return lks

    lakes = generate_random_lakes(num_lakes=num_lakes, r=radius, dim=dim)

    def is_in_region(point):
        return any(np.linalg.norm(point - np.array(center)) < radius for center, radius in lakes)

    y = np.array([-1.0 if is_in_region(p) else 1.0 for p in X])
    y += np.random.normal(scale=0.2, size=y.shape)
    y = torch.Tensor(y)

    return X, y