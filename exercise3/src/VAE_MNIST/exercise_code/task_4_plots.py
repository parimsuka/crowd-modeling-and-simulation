import matplotlib.pyplot as plt
import numpy as np

def plot_reconstructed_test_set(vae, data):
    z_mean, z_log_var, z = vae.encoder(data)
    reconstructed_test_set = vae.decoder(z).numpy()
    
    # Flatten the test set for plotting
    flattened_test_set = data.reshape(-1, 2)
    flattened_reconstructed_set = reconstructed_test_set.reshape(-1, 2)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(flattened_test_set[:, 0], flattened_test_set[:, 1], color='blue', alpha=0.5, label='Original')
    plt.scatter(flattened_reconstructed_set[:, 0], flattened_reconstructed_set[:, 1], color='red', alpha=0.5, label='Reconstructed')
    plt.title('Reconstructed Test Set')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()
    
def plot_generated_samples(vae, latent_dim):
    # Generate random samples from the latent space
    z = np.random.normal(size=(1000, vae.encoder.get_layer("z_mean").output_shape[-1]))
    generated_samples = vae.decoder(z).numpy()
    
#     latent_samples = np.random.normal(size=(1000, latent_dim))
#     generated_samples = vae.decoder.predict(latent_samples)
    
    # Flatten the generated samples for plotting
    flattened_generated_samples = generated_samples.reshape(-1, 2)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(flattened_generated_samples[:, 0], flattened_generated_samples[:, 1], color='green', alpha=0.5, label='Generated')
    plt.title('Generated Samples')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()