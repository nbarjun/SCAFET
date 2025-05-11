# SCAFET: SCAlable Feature Extraction and Tracking

## Overview

SCAlable Feature Extraction and Tracking (SCAFET) is a generalized Python framework designed for the identification of extreme weather events within large-scale weather and climate datasets. This novel method employs curvature measurements of scalar fields to detect and track features of diverse shapes and intensities across various spatial scales, grid types, and dimensions.

Unlike traditional methods that often rely on data-specific, a posteriori physical thresholds, SCAFET uses shape-based thresholds. This approach decouples the feature detection process from inter- and intra-model variability, making it more robust across different datasets and climate scenarios. Furthermore, the time-independent nature of the feature extraction allows for complete parallelization along the time dimension, enhancing computational efficiency and opening possibilities for real-time feature extraction during critical events.

The SCAFET framework is implemented as an easy-to-use, fully open-source Python package, making it accessible even for users with beginner-level Python skills. A simple working example can be found in the accompanying Jupyter Notebook.

## Key Features

* **Shape-Based Detection:** Identifies features based on the curvature of the climate variable field, eliminating the need for arbitrary physical thresholds.
* **Scalable and Versatile:** Detects and tracks features of various shapes and intensities across different scales, grid types (rectilinear and curvilinear), and dimensions (2D and extendable to 3D).
* **Tunable Feature Detection:** The core methodology can be tuned using two key variables: spatial scale and feature shape, allowing for the differentiation of various phenomena (e.g., long filament-shaped atmospheric rivers vs. shorter, round-shaped cyclones).
* **Time-Independent Extraction:** Enables complete parallelization of feature extraction along the time dimension, improving computational efficiency and facilitating parallel pre-processing.
* **Open-Source and User-Friendly:** Fully open-source Python package designed for ease of use, even for users with basic Python knowledge.
* **Modular Design:** Adopts a three-step approach (segmentation, filtering, and tracking) implemented as separate Python libraries, allowing users to substitute individual components with their own methods.

## Core Methodology

SCAFET's novelty lies in its feature detection approach, which avoids a posteriori assumptions and focuses on the overall "shape" of a climate variable field rather than arbitrary thresholding or derivatives. The fundamental detection process remains consistent for various features and can be customized using spatial scale and shape parameters.

The algorithm follows a three-step process:

1.  **Segmentation:** Identifying potential feature regions based on curvature measurements.
2.  **Filtering:** Refining the segmented regions based on user-defined feature properties.
3.  **Tracking:** Linking identified features across time steps to analyze their evolution.

Before these steps, SCAFET requires initialization with the following essential information:

* **Primary Field ($\phi_p$):** The gridded dataset where the target feature is most prominent (e.g., Relative Vorticity for cyclones, Integrated Vapor Transport for atmospheric rivers, SST gradient for SST fronts). Optional secondary fields can be used for further constraints.
* **Grid Properties:** Information about the primary field's grid, including cell area/volume, grid distance, and coastlines, necessary for derivative calculations and landfall identification.
* **Feature Properties:** Characteristics of the target feature, such as estimated spatial scale, shape, eccentricity (for 2D), minimum length, minimum area, minimum volume (for 3D), minimum duration, and maximum distance per time step.

## Usage

For a simple working example and detailed implementation, please refer to the Jupyter Notebook: [scafet\_demo.ipynb](https://github.com/nbarjun/SCAFET/blob/master/scafet_demo.ipynb) (last accessed: 17 October 2023). This notebook provides a practical demonstration of how to use the SCAFET package.

## Applications

This study demonstrates SCAFET's capabilities in detecting various climate features across different grid types, including:

* Atmospheric Rivers
* Cyclones
* Sea Surface Temperature Fronts (SST fronts)
* Jet Streams (in 3D)

While the examples provided focus on these atmospheric and oceanic phenomena, the SCAFET framework can be adapted to detect a wide range of features based on user-defined properties.

## Citation

If you use SCAFET in your research, please cite the following publication:

Nellikkattil, Arjun Babu, Danielle Lemmon, Travis Allen O'Brien, June-Yi Lee, and Jung-Eun Chu. "Scalable Feature Extraction and Tracking (SCAFET): a general framework for feature extraction from large climate data sets." Geoscientific Model Development 17, no. 1 (2024): 301-320.

You can access the publication [here](https://gmd.copernicus.org/articles/17/301/2024/)

## Contact

Dr. Arjun Babu Nellikkattil
arjun.nellikkattil@georgetown.edu
[nbarjun.github.io](https://nbarjun.github.io)