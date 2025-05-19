# RSWNet: A Deep Self-Organizing Map Network for Unsupervised Feature Learning and Clustering
[![Paper](https://img.shields.io/badge/Paper-PDF-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## Overview
RSWNet is a novel deep learning architecture for unsupervised feature learning and clustering. Our approach leverages a modified ResNet50 encoder and a custom decoder architecture for efficient representation learning and clustering, demonstrating superior performance on the [Fashion-MNIST dataset](https://huggingface.co/datasets/zalando-datasets/fashion_mnist).

<p align="center">
  <img src="Images/Model Architecture.png" width="800">
  <br>
  <em>Proposed Method Workflow</em>
</p>

## Key Features
- 🚀 Novel deep learning architecture combining feature learning and clustering
- 📊 Efficient training strategy optimizing both reconstruction and clustering objectives
- 🔍 Comprehensive comparative analysis with traditional clustering methods
- 📈 Superior clustering performance on Fashion-MNIST dataset

## Performance Highlights
| Metric | RSWNet | SOM |
|--------|---------|-----|
| Silhouette Score | 0.8801 | 0.2936 |
| Davies-Bouldin Index | 0.1635 | 0.4945 |
| Calinski-Harabasz Index | 738030.1875 | 3609.6191 |

## Visualizations
The notebook provides 2D visualizations of the learned latent space using both PCA and t-SNE:

PCA of SOM Latents
t-SNE of SOM Latents
These plots help to visually assess the quality of clustering and the separation of classes in the latent space.
### Clustering Visualization
<p align="center">
  <img src="Images/initial_KMeans clustering_raw_data.png" width="800">
  <br>
  <em>Initial K Means Clustering on Raw Training Data (t-SNE)</em>
</p>

<table>
  <tr>
    <td align="center">
      <img src="Images/initial_KMeans clustering_raw_data.png" width="380"><br>
      <em>Initial KMeans Clustering (t-SNE)</em>
    </td>
    <td align="center">
      <img src="Images/Proposed model's K-Means Clustering.png" width="380"><br>
      <em>RSWNet Clustering Results (t-SNE)</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Images/rswnet_latent_pca.png" width="380"><br>
      <em>RSWNet Clustering Results (PCA)</em>
    </td>
    <td align="center">
      <img src="Images/som_clustering.png" width="380"><br>
      <em>SOM Clustering Results (t-SNE)</em>
    </td>
  </tr>
</table>