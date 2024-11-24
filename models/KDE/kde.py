import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from icecream import ic

class KDE:
    def __init__(self, kernel="gaussian"):
        self.kernel = (self.gaussian_kernel if kernel == "gaussian"
                       else self.box_kernel if kernel == "box" else self.triangular_kernel)
        self.data = None
        self.bw = None

    def gaussian_kernel(self, x):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    
    def box_kernel(self, x):
        return np.where(np.abs(x) <= 1, 0.5, 0)
    
    def triangular_kernel(self, x):
        return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

    def fit(self, data):
        self.data = np.atleast_2d(data)

    def select_bandwidth(self, bw_range):
        best_bw = None
        max_likelihood = -np.inf

        for bw in tqdm(bw_range):
            likelihood = 0
            for i in range(self.data.shape[0]):
                one_data = np.delete(self.data, i, axis=0)
                density = self.one_density(self.data[i], one_data, bw)
                if density > 0:
                    likelihood += np.log(density)
            
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                best_bw = bw

        self.bw = best_bw

    def one_density(self, x, data, bw):
        dists = cdist(data, [x]) / bw
        kernel_values = self.kernel(dists).sum(axis=0)
        density = np.clip(kernel_values / (len(data) * bw), 1e-10, np.inf)
        return density

    def eval_density(self, x):
        x = np.atleast_2d(x)
        dists = cdist(self.data, x) / self.bw
        kernel_values = self.kernel(dists)
        density = np.clip(kernel_values.sum(axis=0) / (len(self.data) * self.bw), 1e-10, np.inf)
        return density

    def predict(self, x):
        return self.eval_density(np.atleast_2d(x))

    def visualize(self, x_range=(-4, 4), y_range=(-4, 4), num_points=100, title=None):
        grid_x, grid_y = np.meshgrid(np.linspace(x_range[0], x_range[1], num_points),
                                     np.linspace(y_range[0], y_range[1], num_points))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        density = self.eval_density(grid_points).reshape(num_points, num_points)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(density, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                   origin='lower', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Density')
        plt.title(f'Kernel Density Estimation: {title}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()



