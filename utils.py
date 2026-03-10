######################## utils.py ######################################################################
#                                                                                                     #
# A collection of utility functions for image dataset management, visualization, and model evaluation.#
# Authors: Andrea Loy, Omar Zeroual                                                                   #  
#                                                                                                     #
#######################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def eval_test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).squeeze().long()
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm

def evaluate_test_accuracy(model, test_loader, device):
    # 1. Passage en mode évaluation
    model.eval()
    
    correct = 0
    total = 0
    
    # 2. On désactive le calcul des gradients (important !)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # On récupère l'indice de la classe avec la plus haute probabilité
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted.view(-1) == target.view(-1)).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def display_medmnist_samples(dataset, class_filter=None, n_samples=5):
    """
    Displays images from a MedMNIST dataset object.
    - class_filter: ID (int), Name (str), or None/'*' for all classes.
    - n_samples: Number of images per displayed class.
    """
    X = dataset.imgs
    Y = dataset.labels.flatten()
    label_dict = dataset.info['label']
    
    if class_filter is None or class_filter == '*':
        target_classes = np.unique(Y)
    else:
        if isinstance(class_filter, str):
            match = [int(k) for k, v in label_dict.items() if v.lower() == class_filter.lower()]
            if not match:
                print(f"Class '{class_filter}' not found. Available: {list(label_dict.values())}")
                return
            target_classes = match
        else:
            target_classes = [class_filter]

    n_rows = len(target_classes)
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 3, n_rows * 3.5))
    axes = np.atleast_2d(axes)

    for i, class_id in enumerate(target_classes):
        class_name = label_dict[str(class_id)]
        indices = np.where(Y == class_id)[0]
        n_to_show = min(n_samples, len(indices))
        selected_idx = np.random.choice(indices, n_to_show, replace=False) if n_to_show > 0 else []

        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(selected_idx):
                idx = selected_idx[j]
                ax.imshow(X[idx], cmap='gray' if X[idx].ndim == 2 else None)
                ax.set_title(f"{class_name.upper()}\nIdx: {idx}", fontsize=9, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def print_dataset_structure(dataset):
    """ Prints technical metadata of the MedMNIST dataset. """
    print(f"--- Dataset Structure: {dataset.flag.upper()} ---")
    print(f"Total samples: {len(dataset)}")
    print(f"Image shape: {dataset.imgs.shape}")
    print(f"Labels shape: {dataset.labels.shape}")
    channels = "Grayscale" if dataset.imgs.ndim == 3 else "RGB"
    print(f"Mode: {channels}")

def get_class_stats(dataset):
    """ Internal helper to get a DataFrame of class distribution. """
    labels = dataset.labels.flatten()
    label_dict = dataset.info['label']
    unique, counts = np.unique(labels, return_counts=True)
    
    data = []
    for u, c in zip(unique, counts):
        data.append({
            "ID": u,
            "Class": label_dict[str(u)],
            "Count": c,
            "Percentage": round((c / len(labels)) * 100, 2)
        })
    return pd.DataFrame(data)

def plot_class_distribution(dataset):
    """
    Generates an interactive Plotly bar chart showing class distribution.
    """
    df = get_class_stats(dataset)
    
    fig = px.bar(
        df, 
        x="Class", 
        y="Count",
        text="Percentage",
        color="Class",
        title=f"Class Distribution - {dataset.flag.upper()}",
        labels={"Count": "Number of Samples", "Class": "Medical Category"},
        template="plotly_white"
    )
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    fig.show()

def show_pixel_stats(dataset):

    """
    Displays pixel intensity statistics and distribution (histogram).
    """
    imgs = dataset.imgs
    print(f"--- Pixel Stats: Mean={np.mean(imgs):.2f}, Std={np.std(imgs):.2f}, Min={np.min(imgs)}, Max={np.max(imgs)} ---")
    
    plt.figure(figsize=(10, 4))
    if imgs.ndim == 4: # RGB
        for i, col in enumerate(['red', 'green', 'blue']):
            plt.hist(imgs[:,:,:,i].ravel(), bins=256, color=col, alpha=0.5, label=col)
    else: # Grayscale
        plt.hist(imgs.ravel(), bins=256, color='gray', alpha=0.7)
        
    plt.title("Pixel Intensity Distribution")
    plt.legend()
    plt.show()

def show_random_pixel_stats(dataset, seed=42):
    """
    Selects a random image from the dataset using a seed, 
    displays its metadata, its visual appearance, and its color distribution per channel.
    """
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Select a random index
    random_idx = np.random.randint(0, len(dataset.imgs))
    img = dataset.imgs[random_idx]
    
    print(f"--- Stats for Image Index: {random_idx} (Seed: {seed}) ---")
    print(f"Shape: {img.shape}")
    
    # Calcul et affichage par channel
    if img.ndim == 3: # RGB Image
        channels = ['Red', 'Green', 'Blue']
        for i, color in enumerate(channels):
            channel_data = img[:, :, i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            print(f"{color:5} Channel -> Mean: {mean:.2f} | Std: {std:.2f}")
    else: # Grayscale
        print(f"Grayscale -> Mean: {np.mean(img):.2f} | Std: {np.std(img):.2f}")
    
    # Global stats
    print(f"Global Stats -> Min: {np.min(img)} | Max: {np.max(img)}")

    plt.figure(figsize=(12, 5))
    
    # 1. Visualizing the Image
    plt.subplot(1, 2, 1)
    display_img = img.astype('uint8') if np.max(img) > 1 else img
    plt.imshow(display_img)
    plt.title(f"Image Reference: {random_idx}")
    plt.axis('off')
    
    # 2. Visualizing the Histogram
    plt.subplot(1, 2, 2)
    if img.ndim == 3: # RGB Image
        colors = ('red', 'green', 'blue')
        for i, col in enumerate(colors):
            plt.hist(img[:,:,i].ravel(), bins=256, color=col, alpha=0.5, label=col)
    else: # Grayscale Image
        plt.hist(img.ravel(), bins=256, color='gray', alpha=0.7)
        
    plt.title("Pixel Intensity Distribution per Channel")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plots training and validation loss and accuracy from a history dictionary.
    Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))

    # --- Plot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title('Loss Curve', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Acc', color='#2ca02c', linewidth=2)
    plt.plot(epochs, history['val_acc'], label='Validation Acc', color='#d62728', linestyle='--', linewidth=2)
    plt.axhline(y=55, color='gray', linestyle=':', label='Target (55%)') # Visual target line
    plt.title('Accuracy Curve', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_loader, device, class_names):
    """
    Generates and plots a confusion matrix.
    Returns the CM array and the list of (true, pred) pairs for analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy().flatten())
    
    # Compute Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix: Predicted vs True Class', fontsize=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.show()
    
    return cm