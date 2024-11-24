import matplotlib.pyplot as plt
import numpy as np

def synthetic_data(params):
    num_points_large = params.get("num_points_large", 1000)
    radius_large = params.get("radius_large", 2)
    noise_large = params.get("noise_large", 0.1)

    angles_large = np.random.uniform(0, 2 * np.pi, num_points_large)
    radii_large = radius_large * np.sqrt(np.random.uniform(0, 1, num_points_large))
    x_large = radii_large * np.cos(angles_large) + np.random.normal(0, noise_large, num_points_large)
    y_large = radii_large * np.sin(angles_large) + np.random.normal(0, noise_large, num_points_large)

    num_points_small = params.get("num_points_small", 1000)
    center_small = params.get("center_small", (1, 1))
    radius_small = params.get("radius_small", 0.25)
    noise_small = params.get("noise_small", 0.08)

    angles_small = np.random.uniform(0, 2 * np.pi, num_points_small)
    radii_small = radius_small * np.sqrt(np.random.uniform(0, 1, num_points_small))
    x_small = radii_small * np.cos(angles_small) + center_small[0] + np.random.normal(0, noise_small, num_points_small)
    y_small = radii_small * np.sin(angles_small) + center_small[1] + np.random.normal(0, noise_small, num_points_small)

    x = np.concatenate([x_large, x_small])
    y = np.concatenate([y_large, y_small])

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, color='black', alpha=0.6)
    plt.grid("on")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Original Data")
    plt.savefig("./assignments/5/figures/synthetic_data.png")

    data = np.column_stack([x, y])

    np.save("/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/interim/5/2/synthetic_data.npy", data)


if __name__ == "__main__":
    params = {
        "num_points_large": 3000,
        "radius_large": 2,
        "noise_large": 0.2,
        "num_points_small": 500,
        "center_small": (1, 1),
        "radius_small": 0.25,
        "noise_small": 0.08
    }

    synthetic_data(params)