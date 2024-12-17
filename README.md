# **CM-PHI: A Multi-Component Framework for Predicting Phage-Host Interactions**

This repository contains the implementation of **CM-PHI**, a computational model for predicting phage-host interactions (PHI). CM-PHI integrates graph-based and sequence-based features using multi-hop attention graph neural networks, gated convolutional networks, and self-attention mechanisms.

---

## **Project Overview**

Phage-host interaction prediction is critical for advancing **phage therapy** and understanding microbial ecology. CM-PHI achieves this by:  

1. **Constructing a similarity-based microbial network.**  
2. **Extracting topological features** using Multi-Hop Attention Graph Neural Network (MHAGNN).  
3. **Extracting sequence-level features** using Gated Convolutional Neural Network (GCNN).  
4. **Integrating features** with a Self-Attention Mechanism (SAM).  
5. **Predicting interactions** using a Dual-Input Fusion Network (DIF-Net).

---

## **Project Structure**

The repository is organized as follows:

| **Folder Name**                     | **Description**                                                                               |
|-------------------------------------|-----------------------------------------------------------------------------------------------|
| `Dataset split`                     | Code for splitting the dataset into training, validation, and testing subsets.               |
| `Dual-Input Fusion network`         | Implementation of DIF-Net for feature fusion and PHI classification.                         |
| `Gated Convolutional Networks`      | GCNN module to extract sequence-level features from protein sequences.                       |
| `Multi-hop attention graph neural network` | MHAGNN module for extracting topological features from microbial networks.             |
| `Self-attention mechanism`          | Code for integrating features using the self-attention mechanism (SAM).                      |
| `Similarity calculation`            | Code for calculating cosine similarity to construct microbial similarity networks.           |

---

## **Installation**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/CM-PHI.git
   cd CM-PHI

@article{YourArticle,
  title={A Multi-Component Framework for Predicting Phage-Host Interactions Combining Graph Neural Networks and Sequence Features},
  author={Jie Pan, et al.},
  journal={Your Journal},
  year={2024}
}
