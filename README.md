Graph-Convolutional Autoencoder for Modal Space Analysis
====================================================================

This script supports the paper:  
**"Graph-Convolutional Autoencoder Frameworks for Aerodynamic Shape Predictions of the Agard Wing"**  
Available at: [https://arc.aiaa.org/doi/abs/10.2514/6.2025-0885](https://arc.aiaa.org/doi/abs/10.2514/6.2025-0885)


Purpose:
--------
This module implements a graph-based autoencoder framework (GB-AE-GCN) designed to predict surface aerodynamic fields on parametrised, modal-deformed wing geometries. 
Each sample in the dataset represents a unique combination of six structural mode amplitudes, enabling the model to generalise over a broad family of shape variations.
The proposed architecture integrates:
- Multi-level graph convolution with spatial reduction via Moving Weight Least Squares (MWLS)-based pooling,
- Physics-informed loss function based on moment consistency,
- Bayesian hyperparameter optimisation (via Optuna) for architecture search.

Key Features:
-------------
- Dataset: Surface flow fields including [x, y, z, CP, CFx, CFy, CFz, six modal amplitudes].
- Architecture: Multi-resolution GCN with encoder–decoder structure over reduced grids.
- Loss Function: Combines MAE with physics-informed moment error.
- Output: Surface distributions of aerodynamic coefficients on parametrically deformed shapes.

Dependencies:
-------------
Install the required Python packages before running the scripts:

```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install torch
pip install torch-geometric
```

-------------
Ensure the following input files are available before running the scripts:
- `Dataset\agard_dataset.npy`: steady-state simulation data for $n$ samples of deformed wings;
- `Dataset\grid_data`: folder with volume and surface normal data for all the $n$ samples (4 columns: `X_Grid_K_Unit_Normal`, `Y_Grid_K_Unit_Normal`, `Z_Grid_K_Unit_Normal`, `Cell_Volume`).
- `Dimensionality_reduction\Adjency_matrix\surface.csv`: contains PointIDs of surface mesh nodes;
- `Dimensionality_reduction\Adjency_matrix\mesh.su2`: unstructured mesh file;
- `Dimensionality_reduction\Pressure_gradient\input\dataset_pressure_gradient.npy`: pressure gradient distribution (for the dimensionality reduction module).

Workflow:
---------
1. **Generate adjacency matrix** for the unstructured surface mesh:
   ```bash
   Dimensionality_reduction\Adjency_matrix\write_connectivity_matrix_3D_unstructured.ipynb
````

2. **Run the dimensionality reduction module**:

   ```bash
   Dimensionality_reduction\dimensionality_reduction_module.ipynb
   ```

3. **Run the main script for model generation and prediction**:

   ```bash
   main.ipynb
   ```

Source Files:
-------------
- `Codec.py`, `Connectivity.py`: compute encoder and decoder graphs with Mahalanobis-based edge weighting;
- `MLS_intrp_function.py`: performs MLS-based spatial interpolation;
- `Reduced_space_PDF_function.py`: probabilistic sampling of nodes based on field gradients;
- `space.py`: wrapper for two-level mesh pooling, interpolation, and graph generation;
- `utils.py`: provides auxiliary tools for loading, tracking, and timing.
- `indices_selection_splitting.py`: KMeans-based stratified sampling based on modal coordinates.


Citation
-------------

If you use this code or dataset in your research, please cite:
D. Massegur Sampietro, G. Immordino, A. Vaiuso, A. Da Ronch, and M. Righi.  
Graph-Convolutional Autoencoder Frameworks for Aerodynamic Shape Predictions of the Agard Wing.  
In AIAA SCITECH 2025 Forum, page 0885, 2025.  
https://arc.aiaa.org/doi/abs/10.2514/6.2025-0885


Author: Gabriele Immordino
