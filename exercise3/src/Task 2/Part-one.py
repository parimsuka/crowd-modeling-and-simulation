import numpy as np
import matplotlib.pyplot as plt
from diffusion_map_algo import diffusion_map_algo
from sklearn.datasets import make_swiss_roll

def create_swiss_roll_data(N= 5000):
    """
    Create the swiss roll data for the second part of the task using make swiss roll from
    sklearn.

    Parameters:
    N (int): The number of data points to create.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple of two 1D arrays
    """
    x_k, t_k = make_swiss_roll(N)
    return x_k, t_k


x, t = create_swiss_roll_data(N=1000)


L = 10 #First 10 eigenvalues and eigenvectors
lambda_values, phi = diffusion_map_algo(x, 0.05, L)
for i in range(0, L + 1):
    plt.figure()
    plt.plot(t, phi[:, i])
    plt.title(f"Eigenfunction φ{i} against tk")
    plt.xlabel("tk")
    plt.ylabel(f"φ{i}(xk)")
    plt.grid(True)
    plt.show()


"""
# Generate data set
N = 1000
t_k = np.linspace(0, 2 * np.pi, N, endpoint=False)  # t_k = (2πk)/(N + 1)
X = np.column_stack([np.cos(t_k), np.sin(t_k)])  # x_k = (cos(t_k), sin(t_k))

# Choose appropriate radius and L
radius = 0.009
L = 5

# Apply diffusion map algorithm
lambda_values, phi = diffusion_map_algo(X, radius, L)

# Plot eigenfunctions φl(xk) against tk
for i in range(0, L + 1):
    plt.figure()
    plt.plot(t_k, phi[:, i])
    plt.title(f"Eigenfunction φ{i} against tk")
    plt.xlabel("tk")
    plt.ylabel(f"φ{i}(xk)")
    plt.grid(True)
    plt.show()


N = 1000
t = np.linspace(0, 2*np.pi, N, endpoint=False) # tk = (2πk)/(N + 1)

# xk = (cos(tk), sin(tk)), and then reshape X to be a 1D array
X = np.array([np.cos(t), np.sin(t)]).reshape(-1)

# Computing FFT
fft_result = np.fft.fft(X)

# Extracting the first 5 components and their inverse FFTs (the "eigenfunctions")
eigenvalues = fft_result[:5]
eigenfunctions = [np.fft.ifft(fft_result * (fft_result == ev)) for ev in eigenvalues]

# Plotting
fig, axs = plt.subplots(5, figsize=(10, 20))

for i in range(5):
    axs[i].plot(t, np.real(eigenfunctions[i][:N]))  # Only take the real part, and only plot against the first N points
    axs[i].set_title('Eigenfunction ' + str(i+1))

plt.tight_layout()
plt.show()
"""