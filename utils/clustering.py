import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler #clustering.py
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


class PatchClusterer:
    """Enhanced clustering for grouping recovered patches into images"""

    def __init__(self, num_clusters: int = 8, max_patches_per_cluster: int = 16):
        self.num_clusters = num_clusters
        self.max_patches_per_cluster = max_patches_per_cluster

    def embed_patches(self, patches: List[torch.Tensor]) -> np.ndarray:
        """
        Convert patches to feature embeddings for clustering
        Args:
            patches: List of patch tensors
        Returns:
            Feature matrix for clustering
        """
        embeddings = []

        for patch in patches:
            # Flatten patch to 1D vector
            embedding = patch.flatten().numpy()
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Normalize features
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)

        return embeddings_normalized

    def cluster_patches(self, patches: List[torch.Tensor]) -> Dict[int, List[int]]:
        """
        Cluster patches into groups representing potential images
        Args:
            patches: List of patch tensors
        Returns:
            Dictionary mapping cluster_id to list of patch indices
        """
        if len(patches) == 0:
            return {}

        # Get embeddings
        embeddings = self.embed_patches(patches)

        # Perform clustering
        kmeans = KMeans(n_clusters=min(self.num_clusters, len(patches)),
                        random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group patches by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Limit patches per cluster
        for cluster_id in clusters:
            if len(clusters[cluster_id]) > self.max_patches_per_cluster:
                clusters[cluster_id] = clusters[cluster_id][:self.max_patches_per_cluster]

        return clusters

    def reconstruct_images_from_clusters(self, patches: List[torch.Tensor],
                                         clusters: Dict[int, List[int]],
                                         target_image_size: int = 32) -> List[torch.Tensor]:
        """
        Reconstruct full images from clustered patches
        Args:
            patches: List of all patches
            clusters: Cluster assignments
            target_image_size: Target size for reconstructed images
        Returns:
            List of reconstructed images
        """
        reconstructed_images = []
        patch_size = patches[0].shape[-1]  # Assume square patches
        patches_per_side = target_image_size // patch_size

        for cluster_id, patch_indices in clusters.items():
            # Create canvas for image
            image = torch.zeros(3, target_image_size, target_image_size)

            # Place patches in grid pattern
            for i, patch_idx in enumerate(patch_indices):
                if i >= patches_per_side * patches_per_side:
                    break

                row = i // patches_per_side
                col = i % patches_per_side

                y_start = row * patch_size
                x_start = col * patch_size
                y_end = y_start + patch_size
                x_end = x_start + patch_size

                if patch_idx < len(patches):
                    image[:, y_start:y_end, x_start:x_end] = patches[patch_idx]

            reconstructed_images.append(image)

        return reconstructed_images

    def visualize_clusters(self, patches: List[torch.Tensor],
                           clusters: Dict[int, List[int]], max_patches: int = 32):
        """
        Visualize clustering results
        Args:
            patches: List of patches
            clusters: Cluster assignments
            max_patches: Maximum patches to display
        """
        num_clusters = len(clusters)
        fig, axes = plt.subplots(num_clusters, 8, figsize=(16, num_clusters * 2))

        if num_clusters == 1:
            axes = axes.reshape(1, -1)

        for cluster_id, patch_indices in clusters.items():
            for i, patch_idx in enumerate(patch_indices[:8]):  # Show max 8 patches per cluster
                if patch_idx < len(patches):
                    patch = patches[patch_idx].permute(1, 2, 0).numpy()
                    axes[cluster_id, i].imshow(np.clip(patch, 0, 1))
                    axes[cluster_id, i].set_title(f'C{cluster_id}P{i}')
                    axes[cluster_id, i].axis('off')
                else:
                    axes[cluster_id, i].axis('off')

        plt.suptitle('Clustered Patches')
        plt.tight_layout()
        plt.show()
