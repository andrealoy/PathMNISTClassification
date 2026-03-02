######################## utils.py ######################################################################
#                                                                                                     #
# A collection of utility functions for image dataset management, visualization, and model evaluation.#
# Authors: Andrea Loy, Omar Zeroual                                                                   #  
#                                                                                                     #
#######################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_medmnist_samples(dataset, class_filter=None, n_samples=5):
    """
    Displays images from a MedMNIST dataset object.
    - class_filter: ID (int), Name (str), or None/'*' for all classes.
    - n_samples: Number of images per displayed class.
    """
    X = dataset.imgs
    Y = dataset.labels.flatten()
    label_dict = dataset.info['label']
    
    # 1. Handle filtering logic
    if class_filter is None or class_filter == '*':
        target_classes = np.unique(Y)
    else:
        if isinstance(class_filter, str):
            # Match the name in the dictionary to find the ID
            match = [int(k) for k, v in label_dict.items() if v.lower() == class_filter.lower()]
            if not match:
                print(f"Class '{class_filter}' not found. Available classes: {list(label_dict.values())}")
                return
            target_classes = match
        else:
            target_classes = [class_filter]

    n_rows = len(target_classes)
    # Adjust figure size for titles to breathe
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 3, n_rows * 3.5))
    
    # Ensure axes is a 2D array even for a single row or column
    axes = np.atleast_2d(axes)

    # 3. Filling the grid
    for i, class_id in enumerate(target_classes):
        class_name = label_dict[str(class_id)]
        indices = np.where(Y == class_id)[0]
        
        n_available = len(indices)
        n_to_show = min(n_samples, n_available)
        
        if n_available > 0:
            selected_idx = np.random.choice(indices, n_to_show, replace=False)
        else:
            selected_idx = []

        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(selected_idx):
                idx = selected_idx[j]
                
                # Image display (Auto-detect Grayscale or RGB)
                ax.imshow(X[idx], cmap='gray' if X[idx].ndim == 2 else None)
                
                # Display individual title per image
                ax.set_title(f"{class_name.upper()}\nIdx: {idx}", fontsize=9, fontweight='bold')
            
            # Hide ticks but keep titles visible
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def print_dataset_structure(dataset):
    """
    Prints the technical structure of the MedMNIST dataset.
    """
    print(f"--- Dataset Structure: {dataset.flag.upper()} ---")
    print(f"Object Type: {type(dataset)}")
    print(f"Total samples: {len(dataset)}")
    
    # Image dimensions (N, H, W) or (N, H, W, C)
    print(f"Image format (shape): {dataset.imgs.shape}")
    print(f"Data type (dtype): {dataset.imgs.dtype}")
    
    # Label dimensions
    print(f"Label format: {dataset.labels.shape}")
    
    # Visual mode check
    channels = "Grayscale" if dataset.imgs.ndim == 3 else "RGB (Color)"
    print(f"Visual mode: {channels}")

def show_class_distribution(dataset):
    """
    Displays the statistical distribution of classes in the dataset.
    """
    labels = dataset.labels.flatten()
    label_dict = dataset.info['label']
    
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create a clean table with class names
    dist = []
    for u, c in zip(unique, counts):
        dist.append({
            "ID": u,
            "Class": label_dict[str(u)],
            "Count": c,
            "Percentage": f"{(c/len(labels)*100):.2f}%"
        })
    
    df = pd.DataFrame(dist)
    print("\n--- Class Distribution ---")
    print(df.to_string(index=False))