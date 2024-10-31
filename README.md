### `README.md`

# Penguin Clustering Analysis

This repository contains my solution for a clustering analysis project on penguin species. The project is part of a DataCamp course on Unsupervised Machine Learning, where we utilize clustering techniques to explore and categorize penguin data without labeled species information.

---

## Project Overview

The main goal of this project is to apply clustering methods to distinguish between different penguin species based on morphological measurements. This project helps deepen understanding of clustering algorithms, particularly k-means and hierarchical clustering, in the context of real-world ecological data.

## Dataset

The dataset includes measurements of three penguin species: Adelie, Chinstrap, and Gentoo. Each sample consists of the following features:
- **Culmen Length** (mm)
- **Culmen Depth** (mm)
- **Flipper Length** (mm)
- **Body Mass** (g)
- **Species** (Label used for evaluation)

The dataset is sourced from [Palmer Penguins](https://github.com/allisonhorst/palmerpenguins) and is included in this repository.

---

## Methods

### 1. Data Preprocessing
- **Normalization**: Standardizes the features to improve clustering performance.
- **Missing Data**: Handled missing values by mean imputation.

### 2. Clustering Algorithms
- **K-Means Clustering**: Applied k-means to group penguins into clusters.
- **Evaluation**: Used silhouette score and visual inspection to assess cluster quality.

### 3. Visualization
- **2D Scatter Plots**: Visualized clusters by plotting principal components.
- **Cluster Centers**: Highlighted cluster centers in visualizations to show centroid convergence.

---

## Code Structure

- `sol.py`: Contains the full code for loading, preprocessing, clustering, and visualizing the penguin dataset.

---

## Results and Analysis

The clustering analysis suggests distinct groupings that closely resemble the species divisions in the dataset. The silhouette score and PCA-based plots confirm that the chosen features are effective for clustering purposes.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Required Libraries: Install dependencies via `pip install -r requirements.txt` (not included; add as needed).

### Usage

```bash
# Clone the repository
git clone https://github.com/Aleksandar-Mladenoski/penguin-clustering-ml-project.git

# Navigate to the project directory
cd penguin-clustering-ml-project

# Run the solution script
python sol.py
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

Thanks to DataCamp for providing the project framework and the Palmer Penguins dataset for the data used in this project.
```
https://www.datacamp.com/datalab
```
