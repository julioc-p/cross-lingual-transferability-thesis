# Cross-Lingual Transferability for Text-to-SPARQL Generation

## Project Description

This repository contains the codebase and resources for a Master's Thesis investigating the cross-lingual transferability of Large Language Models (LLMs) for generating SPARQL queries from natural language questions. The project explores how well models pre-trained on different languages can perform this task compared to their base models, and compares different prompting strategies and fine-tuning approaches (referred to as v1 and v2 experiments).

The research involves:
- Acquiring and processing datasets, including variants of the QALD (Question Answering over Linked Data) benchmark.
- Fine-tuning selected LLMs (e.g., Mistral, Occiglot) for the text-to-SPARQL task.
- Evaluating model performance using metrics such as F1-score, Precision, Recall, and Executable Query Percentage.
- Analyzing the impact of different experimental conditions (e.g., providing entity/relationship ID mappings as context) on cross-lingual performance.

The repository includes scripts for data acquisition, data preprocessing, model training, model validation, and results visualization. The final thesis document is also part of this repository.

## Directory Structure

A brief overview of the main directories:

-   `data/`: Contains scripts and notebooks for data acquisition (e.g., from QALD), preprocessing, cleaning, and the resulting datasets.
    -   `data/acquisition/QALD_dataset_generator/`: Scripts to fetch and initially parse QALD datasets.
    -   `data/preprocessing/`: Jupyter notebooks and scripts for cleaning and preparing data for training and validation.
-   `training/`: Scripts and configurations for fine-tuning the LLMs. Organized by model version (v1, v2) and specific model.
-   `validation/`: Includes scripts for evaluating the performance of trained models (e.g., `sparql_validator.py`) and for generating comparative performance plots (e.g., `plot_performance_comparison.py`).
-   `plotting_scripts/`: Contains utility scripts for generating specific plots related to training runs, such as loss curves or gradient norms.
-   `thesis_writing/`: Contains the LaTeX source files for the Master's Thesis document.

## Setup

Follow these steps to set up the project environment:

1.  **Prerequisites:**
    *   Python (version 3.8 or higher recommended).
    *   `pip` (Python package installer).
    *   It is highly recommended to use a Python virtual environment (e.g., `venv` or `conda`) to manage dependencies.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/julioc-p/cross-lingual-transferability-thesis.git
    cd cross-lingual-transferability-thesis
    ```

3.  **Install Dependencies:**
    A unified `requirements.txt` file is provided at the root of the project, with dependencies from various components. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues, individual components (e.g., `data/acquisition/QALD_dataset_generator/`) have their own `requirements.txt` files that were used during development, which you can refer to.*

4.  **Jupyter Notebook Environment:**
    To run the data preprocessing notebooks (e.g., `data/preprocessing/cleaning_sparql_text_dataset.ipynb`), you will need Jupyter Notebook or JupyterLab:
    ```bash
    pip install jupyterlab notebook
    ```

## Workflow: Reproducing the Experiments

The following steps outline the general workflow to reproduce the experiments:

### 1. Data Acquisition

The primary dataset used is derived from varios knowledge graphs including QALD.
-   Navigate to the QALD dataset generator directory:
    ```bash
    cd data/acquisition/QALD_dataset_generator/
    ```
-   Run the main script to fetch and perform initial parsing of the QALD data:
    ```bash
    python main.py
    ```
    This script will download data from specified QALD repositories, process it.

### 2. Data Preprocessing

Once the raw data is acquired, it needs to be cleaned and prepared for training and validation.
-   The main preprocessing and cleaning steps are detailed in the Jupyter Notebook: `data/preprocessing/cleaning_sparql_text_dataset.ipynb`.
-   Launch JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
-   Open and run the cells in `data/preprocessing/cleaning_sparql_text_dataset.ipynb`. This notebook will guide you through the cleaning process, standardization, and potentially splitting the data into training, validation, and test sets. The output will be cleaned datasets ready for model consumption.

### 3. Model Training

Training scripts and configurations are located in the `training/` directory. This directory is organized into subfolders for different experimental versions (e.g., `v1/`, `v2/`) and specific model setups (e.g., `mistral_de_4bit/`, `occiglot_en_4bit/`).

-   Examine the subdirectories within `training/` for specific training scripts (e.g., `train.py` or similar) and configuration files.
-   Execute the relevant scripts within these subdirectories to fine-tune the models. You may need to adjust paths to datasets, model identifiers, and training hyperparameters as per your setup.

### 4. Model Validation and Evaluation

After training, models are evaluated on test sets.
-   The primary script for this is `validation/sparql_validator.py`.
-   This script takes the generated SPARQL queries from your models and compares them against gold standard queries from the test set.
-   It requires the paths to the model outputs and the corresponding gold data.
-   Run the validator script. It will output detailed `summary.txt` files for each evaluated model configuration, containing metrics like Precision, Recall, F1-Score, and Executable Query Percentage.
    ```bash
    # Example
    # cd validation/
    # python sparql_validator.py --model_outputs_dir <path_to_your_model_outputs> --gold_data_dir <path_to_gold_data>
    ```
    Refer to the `sparql_validator.py` script's arguments

### 5. Plotting Results

To visualize and compare the performance of different models and configurations:
-   Navigate to the `validation/` directory:
    ```bash
    cd validation/
    ```
-   Run the `plot_performance_comparison.py` script:
    ```bash
    python plot_performance_comparison.py
    ```
    This script reads the `summary.txt` files generated in the previous step and creates various bar plots (e.g., F1-score, Precision, Recall comparisons) saved as image files (e.g., `f1_score_comparison_subplots.png`) in the `validation/` directory.

-   Additional scripts for plotting specific training metrics (e.g., loss curves, gradient norms from training logs) are available in the `plotting_scripts/` directory. These are typically used for analyzing individual training runs.
    ```bash
    # Example
    # cd plotting_scripts/
    # python generate_v1_1_loss_plot.py --log_file <path_to_training_log>
    ```
    Consult the individual scripts in `plotting_scripts/` for their specific usage and required inputs.

## Thesis Document

The full Master's Thesis document, which details the research, methodology, experiments, and findings, is written in LaTeX.
-   The source files are located in the `thesis_writing/` directory.
-   The main LaTeX file is `thesis_writing/main.tex`.
-   To compile the thesis into a PDF, you will need a standard LaTeX distribution (e.g., TeX Live, MiKTeX, Overleaf). Compile `main.tex`.