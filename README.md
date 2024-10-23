# Continual Learning for Adaptive Neural Decoders

## Overview
**Continual Learning for Adaptive Neural Decoders** is a research project aimed at developing algorithms for real-time analysis and adaptive learning from intracortical recordings. This toolkit is designed to preprocess stimulation data and implement various machine learning models to improve neural decoding performance over time.

The repository is organized to facilitate ease of use, ensuring that preprocessing, modeling, and analysis are straightforward for users.

<div align="center">
  <img src="assets/overview.png" alt="Project Overview" width="110%">
</div>

## Repository Structure
The project is organized into several key directories:

- **Data**: Contains raw data from spinal stimulation recordings in MATLAB format and the preprocessing script.
  - `Preprocessing_Stimulation_Data.py`: Script to preprocess the incoming data.

- **Models**: Houses the architecture of all AI models and stores trained models.
  - `models.py`: Contains the definitions of various neural network architectures.

- **Notebooks**: For data analysis and results visualization.

- **Pyaldata**: Forked repository providing essential modules for data handling.

- **src**: Includes helper functions, regularizers, trainers, and visualizers. 
  - `trainer_hnet_context_infer.py`: Implements the algorithm to detect task identities within the training loop of the hypernetwork.

Outside the folders:
- `config.json`: Configuration file for running experiments.
- `Report_Master_Thesis`: Complete report detailing the findings and methodologies of the Master's Thesis.
- `requirements.txt`: Lists dependencies required to run the project.
- `run.py`: The main function to execute the agnostic continual learning model on neural data.

## Installation

### Prerequisites
Ensure that you have Python 3.x and the necessary libraries installed. You can set up the environment using the `requirements.txt` file.

#### Installation Steps
```bash
pip install -r requirements.txt
```

### Configuration
Edit the `config.json` file to adjust parameters for your experiments, including data paths and model hyperparameters.

### Usage

#### Data Preprocessing
Before training your models, you must preprocess the data using the provided script.

#### Execution
Run the preprocessing script as follows:
```bash
python Data/Preprocessing_Stimulation_Data.py
```

This will prepare the raw MATLAB data for subsequent modeling.

#### Training Models
To run the continual learning model on the preprocessed neural data, execute the main script:
```bash
python run.py
```

This script initializes the training process based on the configurations specified in config.json.

#### Analyzing Results
After training, results can be analyzed using Jupyter notebooks located in the Notebooks folder. These notebooks provide insights into model performance and data interpretation.

#### Expected Output
Upon successful execution, you will find:

Trained models saved in the Models directory.
- Preprocessed data ready for analysis.
- Results visualized in the Notebooks.

#### Contributing
Contributions to enhance the capabilities of this repository are welcome. Please submit issues and pull requests for any enhancements or fixes.
```bash
csharp
```
