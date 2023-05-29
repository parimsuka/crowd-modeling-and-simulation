import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sensitive_area = None

def plot_results(vae, data, range_mask):
    plot_reconstructed_test_set(vae, data, range_mask)
    plot_generated_samples(vae, range_mask)

def plot_reconstructed_test_set(vae, data, range_mask):
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
    
    add_rectangle()
    
    # Set x and y limits to have a distance from 0 to 1
    plt.xlim(-1 if range_mask else 0, 1)
    plt.ylim(-1 if range_mask else 0, 1)
    
    plt.legend()
    plt.show()
    
def plot_generated_samples(vae, range_mask):
    # Generate random samples from the latent space
    z = np.random.normal(size=(1000, vae.encoder.get_layer("z_mean").output_shape[-1]))
    generated_samples = vae.decoder(z).numpy()

    # Flatten the generated samples for plotting
    flattened_generated_samples = generated_samples.reshape(-1, 2)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(flattened_generated_samples[:, 0], flattened_generated_samples[:, 1], color='green', alpha=0.5, label='Generated')
    plt.title('Generated Samples')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    add_rectangle()
    
    # Set x and y limits to have a distance from 0 to 1
    plt.xlim(-1 if range_mask else 0, 1)
    plt.ylim(-1 if range_mask else 0, 1)
    
    plt.legend()
    plt.show()
    
def add_rectangle():
    # Draw a rectangle
    top_left = sensitive_area[0]
    bottom_right = sensitive_area[1]
    width = bottom_right[0] - top_left[0]
    height = top_left[1] - bottom_right[1]
    bottom_left = (top_left[0], top_left[1] - height)
    rectangle = Rectangle(bottom_left, width, height, edgecolor='orange', facecolor='none')
    plt.gca().add_patch(rectangle)
    
def plot_dataset(train_set, test_set):
    plot(train_set)
    plot(test_set)
    
def plot(data):
    add_rectangle()
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    
def normalize_dataset(train_set, test_set, range_mask):
    normalize_sensitive_area(train_set, range_mask)
    train_set = normalize(train_set, range_mask)
    test_set = normalize(test_set, range_mask)
    
    return train_set, test_set

def normalize(dataset, range_mask):
    dataset = np.array(dataset)
    
    # Extract x and y values separately
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    
    # Normalize x values
    x_min = np.min(x_values)
    x_max = np.max(x_values)
    normalized_x = (x_values - x_min) / (x_max - x_min)

    # Normalize y values
    y_min = np.min(y_values)
    y_max = np.max(y_values)
    normalized_y = (y_values - y_min) / (y_max - y_min)
    
    if range_mask:
        normalized_x = normalized_x * 2 - 1
        normalized_y = normalized_y * 2 - 1
    
    # Combine normalized x and y values
    normalized_dataset = np.column_stack((normalized_x, normalized_y))
    return np.expand_dims(normalized_dataset, -1).astype("float32")

def normalize_sensitive_area(dataset, range_mask):
    # 2.6748220028153353 188.49331237472774
    # 4.683671998181228 108.17577426393086
    global sensitive_area 
    sensitive_area = np.expand_dims([[130, 70], [150, 50]], -1).astype("float32")
    
    # Extract x and y values separately
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    
    # Normalize x values
    x_min = np.min(x_values)
    x_max = np.max(x_values)
    # Normalize y values
    y_min = np.min(y_values)
    y_max = np.max(y_values)
    

    x_values = sensitive_area[:, 0]
    y_values = sensitive_area[:, 1]
    sensitive_area_x = (x_values - x_min) / (x_max - x_min)
    sensitive_area_y = (y_values - y_min) / (y_max - y_min)
    
    if range_mask:
        sensitive_area_x = sensitive_area_x * 2 - 1
        sensitive_area_y = sensitive_area_y * 2 - 1
    
    sensitive_area = np.column_stack((sensitive_area_x, sensitive_area_y))

def is_inside_sensitive_area(position):
    """
    Check if a given position falls inside the sensitive area.

    Args:
        position (list): The position coordinates [x, y].
        sensitive_area (list): The coordinates of the sensitive area
                               [[x1, y1], [x2, y2]].

    Returns:
        bool: True if the position is inside the sensitive area, False otherwise.
    """
    x, y = position
    x1, y1 = sensitive_area[0]
    x2, y2 = sensitive_area[1]

    if x >= x1 and x <= x2 and y >= y2 and y <= y1:
        return True
    else:
        return False