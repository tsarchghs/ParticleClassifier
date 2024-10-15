# Particle Classification and Analysis

This project analyzes particle distributions in images, focusing on classifying red and green particles. The workflow includes data generation, feature extraction, hypothesis testing, and training machine learning models using stacking classifiers.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Generation](#data-generation)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Hypothesis Testing](#hypothesis-testing)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Generate synthetic images of red and green particles.
- Extract key features from images, such as:
  - Count of red and green pixels.
  - Red-to-green pixel ratio.
  - Average distances between particles.
  - Vertical and horizontal pixel distributions.
- Perform statistical hypothesis testing on particle distributions.
- Train and evaluate models using a stacking classifier approach.
- Save trained models and scalers for future use.

## Installation

Ensure you have Python 3.7 or higher installed. Clone this repository and install the required packages using the following commands:

```bash
git clone https://github.com/tsarchghs/ParticleClassifier.git
cd ParticleClassifier
pip install -r requirements.txt
```
Potential Applications
Biological Studies: Analyzing cell distributions in biological samples.
Material Science: Understanding the distribution of different particle types in composites or nanomaterials.
Environmental Monitoring: Assessing pollutant distribution in air or water samples.
