import matplotlib.pyplot as plt
import numpy as np

# Define the plot loss function

def plot_losses(h):
    plt.plot(h['loss'])
    plt.plot(h['reconstruction_loss'])
    plt.title('Total Loss & Reconstruction Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['L', 'RL'], loc='upper left')
    plt.show()

    plt.plot(h['kl_loss'])
    plt.title('KL Divergence Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['KL'], loc='upper left')
    plt.show()
    
    
# Define the 2D plot latent manifold function

def plot_latent_manifold(vae):
    # Define the grid in the latent space
    n = 15  # Number of digits along each axis
    digit_size = 28  # Size of each digit image
    scale = 2.0  # Scale factor for spacing between digits
    figsize = 10  # Size of the figure

    # Create a grid of coordinates in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]  # Reverse order for better visualization
    latent_grid = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    # Generate digits from the latent space grid
#     generated_digits = decoder.predict(latent_grid)
    generated_digits = vae.decoder(latent_grid).numpy()

    # Reshape and plot the generated digits
    plt.figure(figsize=(figsize, figsize))
    for i, digit in enumerate(generated_digits):
        ax = plt.subplot(n, n, i + 1)
        plt.imshow(digit.reshape(digit_size, digit_size), cmap='gray')
        plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    

# Plot latent space of all digits

def plot_latent_space(vae, data, labels):
    z_mean, _, _ = vae.encoder(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='jet')
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Representation")
    plt.colorbar()
    plt.show()
    
    
# Plot reconstruction of n digits

def plot_reconstructions(vae, data, n=15):
    z_mean, z_log_var, z = vae.encoder(data)
    reconstructions = vae.decoder(z).numpy()
    
    # Select random samples
#     sample_indices = np.random.randint(0, len(x_test), size=15)
#     original_digits = data[sample_indices]
#     reconstructions = reconstructions[sample_indices]

    original_digits = data
    
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_digits[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Reconstructed Digits")
    plt.show()
    
    
# Plot n generated digits sampled from p(x)

def plot_generated(vae, n=15):
    z = np.random.normal(size=(n, vae.encoder.get_layer("z_mean").output_shape[-1]))
    generated = vae.decoder(z).numpy()
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(generated[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Generated Digits")
    plt.show()
    
    
# Call all plot functions

def plot_all(vae, data, labels, history):
    print("Plotting all losses")
    plot_losses(history)
    
    print("Plotting latent space")
    plot_latent_space(vae, data, labels)
    
    print("Plotting reconstructions")
    plot_reconstructions(vae, data)
    
    print("Plotting generated digits")
    plot_generated(vae)
    
    print("Plotting 2D manifold")
    plot_latent_manifold(vae)