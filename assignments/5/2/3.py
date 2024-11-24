from models.KDE.kde import KDE
import numpy as np
import matplotlib.pyplot as plt
from models.GMM.GMM import GMM
from scipy.stats import multivariate_normal


data = np.load("/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/interim/5/2/synthetic_data.npy")

kde = KDE(kernel="triangular") 
kde.fit(data)
kde.select_bandwidth(np.linspace(0.1, 1.0, 100))

Xg, Yg = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
grid = np.array([Xg.ravel(), Yg.ravel()]).T

kde_density = kde.eval_density(grid).reshape(Xg.shape)

gmm = GMM(num_comp=2)
gmm.fit(data)

gmm_density = np.zeros(grid.shape[0])
for k in range(gmm.num_comp):
    rv = multivariate_normal(mean=gmm.means[k], cov=gmm.cov[k])
    gmm_density += gmm.W[k] * rv.pdf(grid)
gmm_density = gmm_density.reshape(Xg.shape)


def visualize_density_comparison(data, kde_density, gmm_density, x_range=(-4, 4), y_range=(-4, 4), num_points=100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(data[:, 0], data[:, 1], s=2, color='black', alpha=0.5)
    axes[0].set_title("Original Data")
    axes[0].set_xlim(x_range)
    axes[0].set_ylim(y_range)

    im1 = axes[1].imshow(gmm_density, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                         origin='lower', cmap='viridis', interpolation='nearest')
    axes[1].set_title("GMM Density")
    axes[1].set_xlim(x_range)
    axes[1].set_ylim(y_range)
    fig.colorbar(im1, ax=axes[1], orientation="vertical", label="Density")

    im2 = axes[2].imshow(kde_density, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                         origin='lower', cmap='viridis', interpolation='nearest')
    axes[2].set_title("KDE Density")
    axes[2].set_xlim(x_range)
    axes[2].set_ylim(y_range)
    fig.colorbar(im2, ax=axes[2], orientation="vertical", label="Density")

    plt.tight_layout()
    plt.savefig("./assignments/5/figures/comparison.png")


if __name__ == "__main__":
    visualize_density_comparison(data, kde_density, gmm_density, x_range=(-4, 4), y_range=(-4, 4), num_points=100)


