import pandas as pd
from sklearn.datasets import load_wine
from sklearn.datasets import make_regression, make_friedman1, make_sparse_uncorrelated, make_blobs, make_circles, make_moons
import matplotlib.pyplot as plt

def draw_plot(X_value, y_value, title, output_file):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.scatter(X_value, y_value, color='blue', s=100, alpha=0.5)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Target')
    ax.set_title(title)

    # draw plot
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(output_file)
    plt.close(fig)

# toy dataset
dsWine = load_wine()
#print(dsWine.data[:5])

# customers fake dataset
dsCustom = pd.read_csv("data/customers.csv")
print(dsCustom.head(5))

# generate regression dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=42)
draw_plot(X[:, 0], y, 'Regression dataset', 'data/out/plot_regression.png')

# generate friedman dataset
X_friedman, y_friedman = make_friedman1(n_samples=100, n_features=10, noise=1.0, random_state=42)
draw_plot(X_friedman[:, 0], y_friedman, 'Friedman dataset', 'data/out/plot_friedman.png')

# generate sparse dataset
X_sparse, y_sparse = make_sparse_uncorrelated(n_samples=100, n_features=10, random_state=42)
draw_plot(X_sparse[:, 0], y_sparse, 'Sparse dataset', 'data/out/plot_sparse.png')

X_blobs, y_blobs = make_blobs(n_samples=100, centers=4, random_state=42)
draw_plot(X_blobs[:, 0], y_blobs, 'Blobs dataset', 'data/out/plot_blobs.png')

X_circles, y_circles = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
draw_plot(X_circles[:, 0], y_circles, 'Circles dataset', 'data/out/plot_circles.png')

X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)
draw_plot(X_moons[:, 0], y_moons, 'Moons dataset', 'data/out/plot_moons.png')