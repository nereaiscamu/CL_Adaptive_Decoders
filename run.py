### Imports
import pandas as pd
import numpy as np
import pickle
import json
import argparse
import os
import sys

# Importing necessary deep learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import *
import torch.utils.data as data
from torch.utils.data import Dataset
from hypnettorch.hnets import HyperNetInterface
from hypnettorch.hnets import HMLP
import copy
import time

# Set the current directory and navigate up to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..',))
sys.path.append(parent_dir)

# Importing necessary modules from local source files
from src.trainer_hnet_context_infer import *
from src.helpers import *
from Models.models import *
from src.helpers_task_detector import *

# Set device for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Class to configure experiment parameters
class ExperimentConfig:
    """
    A class to manage the configuration settings for an experiment in the context of deep learning.

    Attributes:
    hidden_units (int): The number of hidden units in the neural network.
    num_layers (int): The number of layers in the neural network.
    dropout (float): The dropout rate for regularization.
    lr_detector (float): The learning rate for the task detector.
    lr_step_size (int): Step size for learning rate scheduling.
    lr_gamma (float): Multiplicative factor for learning rate decay.
    seq_length_LSTM (int): The sequence length for LSTM inputs.
    batch_size_train (int): Batch size for training the model.
    batch_size_val (int): Batch size for validation.
    delta (float): Threshold parameter for certain algorithms.
    l1_ratio_reg (float): L1 regularization ratio.
    alpha_reg (float): Alpha value for regularization.
    lr_hnet (float): Learning rate for the hypernetwork.
    beta_hnet_reg (float): Regularization strength for the hypernetwork.
    thrs (float): Threshold for model evaluation or decision making.
    hidden_layers_hnet (list): Configuration for hidden layers in the hypernetwork.
    emb_size (int): Size of the embedding layer.
    experiment_name (str): Name identifier for the experiment.
    """

    def __init__(self, experiment):
        """
        Initializes an instance of the ExperimentConfig class with specified parameters.

        Args:
        experiment (dict): A dictionary containing configuration parameters for the experiment.
        """
        self.hidden_units = experiment['hidden_units']
        self.num_layers = experiment['num_layers']
        self.dropout = experiment['dropout']
        self.lr_detector = experiment['lr_detector']
        self.lr_step_size = experiment['lr_step_size']
        self.lr_gamma = experiment['lr_gamma']
        self.seq_length_LSTM = experiment['seq_length_LSTM']
        self.batch_size_train = experiment['batch_size_train']
        self.batch_size_val = experiment['batch_size_val']
        self.delta = experiment['delta']
        self.l1_ratio_reg = experiment['l1_ratio_reg']
        self.alpha_reg = experiment['alpha_reg']
        self.lr_hnet = experiment['lr_hnet']
        self.beta_hnet_reg = experiment['beta_hnet_reg']
        self.thrs = experiment['thrs']
        self.hidden_layers_hnet = experiment['hidden_layers_hnet']
        self.emb_size = experiment['embedding_size']
        self.experiment_name = experiment['experiment_name']

# Class to run the experiment for Block 3
class Run_Experiment_Block3:

    """
    A class to manage the execution of experiments, particularly focusing on Block 3 experiments in continual learning.

    Attributes:
    config (ExperimentConfig): Configuration settings for the experiment.
    device (torch.device): The device (CPU or GPU) used for computations.
    datasets (dict): A dictionary containing training, validation, and test datasets.
    num_features (int): Number of features in the input dataset.
    num_dim_output (int): Number of output dimensions from the model.
    num_conditions (int): Number of conditions for the task detector.
    LSTM (bool): Flag indicating whether to use LSTM or RNN.
    hnet (HMLP): Hypernetwork model used for task-specific weight generation.
    model (RNN_Main_Model): Main model used for predictions.
    n_contexts (int): The current number of contexts/tasks encountered.
    continual_trainer (ContinualLearningTrainer): Trainer responsible for managing continual learning.
    calc_reg (bool): Flag to determine whether to calculate regularization during training.
    """


    def __init__(self, config, device, datasets, LSTM):

        """
        Initializes an instance of the Run_Experiment_Block3 class.

        Args:
        config (ExperimentConfig): Configuration settings for the experiment.
        device (torch.device): The device (CPU or GPU) used for computations.
        datasets (dict): A dictionary containing training, validation, and test datasets.
        LSTM (bool): Flag indicating whether to use LSTM or RNN for the task detector.
        """


        self.config = config
        self.device = device
        self.datasets = datasets
        self.num_features = datasets[list(datasets.keys())[0]][0].shape[2]
        self.num_dim_output = datasets[list(datasets.keys())[0]][1].shape[2]
        self.num_conditions = 60
        self.LSTM = LSTM
        self.hnet = self._initialize_hnet()  # Initialize Hypernetwork
        self.model = self._initialize_model()  # Initialize Main Model
        self.n_contexts = 0
        self.continual_trainer = ContinualLearningTrainer(self.model, self.hnet, self.n_contexts, self.device)
        self.calc_reg = False  # Flag to calculate regularization

    def _initialize_hnet(self):
        """
        Initializes the Hypernetwork based on the parameters from the task detector.

        Returns:
        hnet (HMLP): An instance of the HyperMLP class configured for the task.
        """

        param_shapes = [p.shape for p in list(self._initialize_task_detector().parameters())]
        hnet = HMLP(param_shapes, uncond_in_size=0,
                    cond_in_size=self.config.emb_size,
                    layers=self.config.hidden_layers_hnet,
                    num_cond_embs=self.num_conditions).to(self.device)

        # Enable gradient computation for Hypernetwork parameters
        for param in hnet.parameters():
            param.requires_grad = True
        hnet.apply_hyperfan_init()  # Initialize weights with Hyperfan initialization
        return hnet

    def _initialize_task_detector(self):

        """
        Initializes the appropriate task detector (LSTM or RNN) based on the configuration settings.

        Returns:
        task_detector (Causal_Simple_LSTM or Causal_Simple_RNN): An instance of the selected task detector class.
        """

        if self.LSTM:
            return Causal_Simple_LSTM(
                num_features=self.num_features, 
                hidden_units=self.config.hidden_units, 
                num_layers=self.config.num_layers, 
                out_dims=self.num_dim_output,
                dropout=self.config.dropout
            ).to(self.device)
        else:
            return Causal_Simple_RNN(
                num_features=self.num_features, 
                hidden_units=self.config.hidden_units, 
                num_layers=self.config.num_layers, 
                out_dims=self.num_dim_output,
                dropout=self.config.dropout
            ).to(self.device)
        


    def _initialize_model(self):
        """
        Initializes the main model with output weights generated by the Hypernetwork.

        Returns:
        model (RNN_Main_Model): An instance of the main model configured for the experiment.
        """
        w_test = self.hnet(cond_id=0)
        model = RNN_Main_Model(
            num_features=self.num_features, 
            hnet_output=w_test,  
            hidden_size=self.config.hidden_units,
            num_layers=self.config.num_layers,
            out_dims=self.num_dim_output,  
            dropout=self.config.dropout,  
            LSTM_=self.LSTM
        ).to(self.device)

        # Disable gradient computation for model parameters
        for param in model.parameters():
            param.requires_grad = False
        return model
    

    def evaluate_model(self, x_train, y_train, x_val, y_val, x_test, y_test, model):
        """
        Evaluates the model using the R^2 metric.

        Args:
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        x_val (torch.Tensor): Validation input data.
        y_val (torch.Tensor): Validation target data.
        x_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test target data.
        model (RNN_Main_Model): The model to be evaluated.

        Returns:
        tuple: Predictions and scores for training, validation, and test sets.
        """
        y_hat, y_true, train_score, v_score, test_score = eval_model(
            x_train, y_train,
            x_val, y_val,
            x_test, y_test, 
            model, 
            metric='r2'
        )
        return y_hat, y_true, train_score, v_score, test_score
    


    def train_hnet(self, x_train, y_train, x_val, y_val, task_id, calc_reg):
        """
        Trains the Hypernetwork for the current task, managing training parameters and returning results.

        Args:
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        x_val (torch.Tensor): Validation input data.
        y_val (torch.Tensor): Validation target data.
        task_id (int): Identifier for the current task being trained.
        calc_reg (bool): Flag indicating whether to apply regularization.

        Returns:
        dict: Results including loss and validation scores for the current training iteration.
        """
        best_model, train_losses, val_losses, change_detect_epoch, \
        prev_context, prev_min_loss, prev_mean_loss, new_context, \
        new_min_loss, new_mean_loss, sim_score, active_context = self.continual_trainer.train_current_task(
            y_train, 
            x_train, 
            y_val,
            x_val, 
            calc_reg=calc_reg,
            cond_id=int(task_id),
            lr=self.config.lr_hnet,
            lr_step_size=5,
            lr_gamma=self.config.lr_gamma,
            sequence_length_LSTM=self.config.seq_length_LSTM,
            batch_size_train=self.config.batch_size_train,
            batch_size_val=self.config.batch_size_train,
            num_epochs=40,  # Adjust the number of epochs as needed
            delta=self.config.delta,
            beta=self.config.beta_hnet_reg, 
            regularizer=reg_hnet,
            l1_ratio=self.config.l1_ratio_reg,
            alpha=0.01,
            early_stop=5,
            chunks=False, 
            LSTM_=self.LSTM
        )
        return best_model, train_losses, val_losses, change_detect_epoch, \
               prev_context, prev_min_loss, prev_mean_loss, new_context, \
               new_min_loss, new_mean_loss, sim_score, active_context

def run(self):
    """
    Execute the training and evaluation of the Hypernetwork across multiple datasets.

    This method trains the Hypernetwork for each dataset, computes the explained variance,
    and saves the training results and model weights.

    Returns:
        dict: A dictionary containing results for each dataset.
    """
    results_dict = {}  # Dictionary to store results for each dataset
    seen_sets = 0  # Counter for seen datasets

    # Iterate over all datasets
    for dataset_name in self.datasets.keys():
        # Initialize a dictionary to store results for the current dataset
        results_dict_subset = {}
        
        # Unpack the dataset into training, validation, and test sets
        x_train, y_train, x_val, y_val, x_test, y_test = self.datasets[dataset_name]

        # Create a directory to save Hypernetwork models
        path_hnet_models = f'./Models/Models_HNET_Block3/{self.config.experiment_name}'
        os.makedirs(path_hnet_models, exist_ok=True)  # Ensure the directory exists

        start_time = time.time()  # Start timer for training
        print(f"Running dataset: {dataset_name}")

        # Determine if regularization should be calculated based on active contexts
        self.calc_reg = self.continual_trainer.n_contexts >= 1
        print(f"{dataset_name}: Calculate regularization: {self.calc_reg}")

        # Train the Hypernetwork and capture various outputs
        (self.hnet, train_losses_, val_losses_,
         change_detect_epoch, prev_context, prev_min_loss,
         prev_mean_loss, new_context, new_min_loss,
         new_mean_loss, sim_score, active_context) = self.train_hnet(
            x_train, y_train, x_val, y_val, 
            self.continual_trainer.active_context, 
            calc_reg=self.calc_reg
        )
        
        print(f'Number of contexts: {self.continual_trainer.n_contexts}')

        # Calculate explained variance using the best weights from the Hypernetwork
        W_best = self.hnet(cond_id=self.continual_trainer.active_context)
        r2_test, y_pred_test = calc_explained_variance_mnet(x_test, y_test, W_best, self.model)

        # Store true and predicted values, and explained variance in the results dictionary
        results_dict_subset['y_true_hnet'] = y_test
        results_dict_subset['y_pred_hnet'] = y_pred_test
        results_dict_subset['r2_test_hnet'] = r2_test

        # Generate model ID for saving
        model_id = f"{self.continual_trainer.active_context}_{seen_sets}"
        save_model(self.hnet, model_id, path_hnet_models)  # Save model weights

        # Store training and evaluation results
        results_dict_subset.update({
            'hnet_train_losses': train_losses_,
            'hnet_val_losses': val_losses_,
            'training_time': time.time() - start_time,
            'change_detect_epoch': change_detect_epoch,
            'prev_active_context': prev_context,
            'prev_min_loss': prev_min_loss,
            'prev_mean_loss': prev_mean_loss,
            'new_tested_context': new_context,
            'new_loss': new_min_loss,
            'new_mean_loss': new_mean_loss,
            'sim_score': sim_score,
            'active_context': active_context
        })

        results_dict[dataset_name] = results_dict_subset  # Store results for the current dataset
        seen_sets += 1  # Increment seen sets count

    return results_dict  # Return the collected results

    

def main(args):

    """
    Main function to run experiments based on configurations defined in a JSON file.

    Args:
        args: Command-line arguments containing the index of the experiment to run 
              and a flag for sorting data.
    """
     
    index = args.index
    sort = bool(args.sort)

    # Load the list of experiments from JSON
    with open(os.path.join('config.json'), 'r') as f:
        experiments = json.load(f)


    LSTM_mode = True # Set LSTM mode
    
    # Run experiments either for a specified index or for a range of experiments
    if index == -1:
        # Run experiments for the specified range (160 to 171)
        for exp in range(160, 172):
            run_experiment(experiments[exp], sort, LSTM_mode)
    else:
        # Run a single experiment based on the provided index
        run_experiment(experiments[index], sort, LSTM_mode)


def run_experiment(experiment, sort, LSTM_mode):
    """
    Run a single experiment with the given configuration.

    Args:
        experiment: The experiment configuration dictionary.
        sort: Boolean indicating whether to sort the data.
        LSTM_mode: Boolean indicating whether to use LSTM.
    """
    name = experiment['experiment_name']
    print(f'Running experiment: {name}')

    # Load dataset
    data = experiment['data']
    data_dir = "./Data/"
    with open(os.path.join(data_dir, f'{data}.pkl'), 'rb') as fp:
        sets = pickle.load(fp)
    print('Data found')

    # Sort the data if requested
    if sort:
        print('Sorting the data...')
        num_trials = experiment['num_trials']
        sets = create_sets(sets, num_trials)  # Sort or trim the data
        # Save the sorted dataset for future reference
        path_to_save_data = os.path.join(data_dir, f'{data}_{num_trials}trials_v4.pkl')
        with open(path_to_save_data, 'wb') as handle:
            pickle.dump(sets, handle, protocol=4)
        print("Data saved.")

    # Run the experiment
    print('Running experiment...')
    config = ExperimentConfig(experiment)
    runner = Run_Experiment_Block3(config, device, sets, LSTM=LSTM_mode)  # Change to true if needed
    results_dict = runner.run()

    # Save results to a file
    save_results(name, results_dict)

def save_results(name, results_dict):
    """
    Save the experiment results to a pickle file.

    Args:
        name: The name of the experiment.
        results_dict: Dictionary containing the results of the experiment.
    """
    path_to_results = os.path.join('.', 'Results')
    os.makedirs(path_to_results, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(path_to_results, f'{name}.pkl')
    
    # Save the results dictionary using pickle
    with open(file_path, 'wb') as fp:
        pickle.dump(results_dict, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script to run experiments")
    
    parser.add_argument(
        "--index",
        type=int,
        default=0,  # Default to 0 to run the first experiment on the list
        help="Index to iterate over the experiment configurations",
    )
    
    parser.add_argument(
        "--sort",
        type=int,
        default=0,
        help="If data needs to be sorted to get the baseline first",
    )

    args = parser.parse_args()
    main(args)   