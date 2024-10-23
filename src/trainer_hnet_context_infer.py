import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import *
from copy import deepcopy
import torch.utils.data as data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import math

# Import custom modules
from src.regularizers import *
from src.trainer import *
from src.helpers import *
from Models.models import *
from src.sequence_datasets import *

class ContinualLearningTrainer:

    """
    A trainer class for managing continual learning tasks with hypernetworks.
    
    Attributes:
        model: The main model used for predictions.
        hnet: The hypernetwork that generates weights for the main model.
        n_contexts: The number of contexts/tasks the model can learn from.
        device: The device on which computations are performed (e.g., 'cuda').
        new_context: Flag indicating if a new context has been detected.
        context_error_0: Initial context error parameter.
        context_error: List to store context errors for each context.
        confidence_context: List to store confidence levels for each context.
        active_context: Index of the currently active context.
        thresholds_contexts: Thresholds for detecting context changes.
        task_covariances: List of covariance matrices for each context.
        task_cov_counts: Counts of updates for each context's covariance.
        rolling_covariances: Store the last covariance matrices for moving average calculations.
    """
        

    def __init__(self, model, hnet, n_contexts, device='cuda'):
        """
        Initializes the ContinualLearningTrainer with the provided model and hypernetwork.
        
        Args:
            model: The main model for prediction.
            hnet: The hypernetwork generating weights for the model.
            n_contexts: Maximum number of contexts (tasks) the trainer will handle.
            device: Device for computation (default is 'cuda').
        """
        self.model = model
        self.hnet = hnet
        self.device = device
        self.n_contexts = n_contexts
        self.new_context = False
        
        # Initialize context errors and confidence
        self.context_error_0 = torch.nn.Parameter(torch.zeros((1)), requires_grad=False)
        self.context_error = [self.context_error_0]
        self.confidence_context = [0]
        self.active_context = 0
        self.thresholds_contexts = torch.nn.Parameter(torch.full((60,), 2.2), requires_grad=False)  # Context thresholds

        # Initialize covariance matrices and counts for each context
        self.task_covariances = []
        self.task_cov_counts = []
        self.rolling_covariances = []  # Store the last covariance matrices

    def deviate_from_mean(self, modulation, context):
        """
        Checks if the current modulation deviates significantly from the mean error
        of the specified context.
        
        Args:
            modulation: The current modulation loss value.
            context: The index of the context to check against.
        
        Returns:
            bool: True if the modulation deviates significantly from the mean; False otherwise.
        
        Raises:
            IndexError: If context index is out of range.
            ValueError: If the mean value is zero.
        """
        N = 100
        k = 15
        
        # Validate context index
        if not (0 <= context < len(self.context_error)):
            raise IndexError(f"Context index {context} is out of range.")
        
        context_errors = self.context_error[context]
        
        # Compute metrics for deviation check
        min_loss = torch.min(context_errors[-k:-1].min(), modulation)
        bar = torch.mean(context_errors[-N:-1])
        
        if bar.item() == 0:  # Avoid division by zero
            raise ValueError("Mean value (bar) is zero, cannot divide by zero.")
        
        # Return whether the deviation exceeds the threshold
        return min_loss / bar > 1.4  # Adjust threshold for stimulation data

    
    
    def update_rolling_covariance(self, features):
        """
        Updates the rolling covariance with new feature data.
        
        Args:
            features: The new feature data used to calculate covariance.
        """
        new_covariance = compute_covariance_matrix(features)
        self.rolling_covariances.append(new_covariance)
        if len(self.rolling_covariances) > 20:
            self.rolling_covariances.pop(0)  # Maintain a fixed size

    def update_covariance_for_context(self, context):
        """
        Updates the mean covariance for a specific context.
        
        Args:
            context: The index of the context to update.
        """
        rolling_mean_covariance = self.compute_mean_covariance(self.rolling_covariances)

        if len(self.task_covariances) <= context:
            self.task_covariances.append(rolling_mean_covariance)
            self.task_cov_counts.append(1)
        else:
            self.task_covariances[context] = update_mean_covariance(
                self.task_covariances[context],
                rolling_mean_covariance,
                self.task_cov_counts[context]
            )
            self.task_cov_counts[context] += 1



    def compute_mean_covariance(self, covariances):
        """
        Calculates the mean of a list of covariance matrices.
        
        Args:
            covariances: List of covariance matrices to average.
        
        Returns:
            Tensor: The mean covariance matrix or None if the list is empty.
        """
        return sum(covariances) / len(covariances) if covariances else None

    
    
    def is_similar_to_previous_tasks(self, current_covariance, threshold=5):
        """
        Checks if the current task is similar to any previously learned tasks 
        based on covariance metrics.
        
        Args:
            current_covariance: The covariance matrix of the current task.
            threshold: Similarity threshold for comparison.
        
        Returns:
            tuple: (bool, int) indicating if a similar task was found, 
                   and the index of the most similar task (or -1 if none).
        """
        similar_tasks = []

        for i, prev_covariance in enumerate(self.task_covariances):
            similarity_score = torch.abs(current_covariance - prev_covariance).mean().item()
            if similarity_score < threshold:
                similar_tasks.append((i, similarity_score))

        if similar_tasks:
            # Return the most similar task
            similar_tasks.sort(key=lambda x: x[1])
            return True, similar_tasks[0][0]

        return False, -1  # No similar tasks found
    

    def train_current_task(self, y_train, x_train, y_val, x_val, **kwargs):
        """
        Trains the model on the current task using provided training and validation data.
        
        Args:
            y_train: Training labels.
            x_train: Training features.
            y_val: Validation labels.
            x_val: Validation features.
            **kwargs: Hyperparameters for training such as learning rate, batch size, etc.
        """

        # Unpack hyperparameters from kwargs
        lr = kwargs.get('lr', 0.0001)
        lr_step_size = kwargs.get('lr_step_size', 10)
        lr_gamma = kwargs.get('lr_gamma', 0.9)
        sequence_length_LSTM = kwargs.get('sequence_length_LSTM', 15)
        batch_size_train = kwargs.get('batch_size_train', 25)
        batch_size_val = kwargs.get('batch_size_val', 25)
        num_epochs = kwargs.get('num_epochs', 1000)
        delta = kwargs.get('delta', 8)
        beta = kwargs.get('beta', 0)
        regularizer = kwargs.get('regularizer', None)
        l1_ratio = kwargs.get('l1_ratio', 0.5)
        alpha = kwargs.get('alpha', 1e-5)
        early_stop = kwargs.get('early_stop', 10)
        LSTM_ = kwargs.get('LSTM_', False)

        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(self.hnet.internal_params, lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        best_model_wts = None
        best_loss = float('inf')
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

        # Create datasets and data loaders
        train_dataset = SequenceDataset(y_train, x_train, sequence_length=sequence_length_LSTM)
        val_dataset = SequenceDataset(y_val, x_val, sequence_length=sequence_length_LSTM)
        loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

        # Initialize hidden states for LSTM if specified
        hx = self.initialize_hidden_states(batch_size_train, LSTM_)

        if self.n_contexts == 0:
            self.n_contexts += 1

        prev_hnet = deepcopy(self.hnet)

        # Generate empty lists to store results
        train_losses = []
        val_losses = []
        change_detect_epoch = []
        prev_context = []
        prev_min_loss = []
        prev_mean_loss = []
        new_context = []
        new_min_loss = []
        new_mean_loss = []
        similarity_scores = []

        # Main training loop
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                self.hnet.train() if phase == 'train' else self.hnet.eval()
                loader = loader_train if phase == 'train' else loader_val

                running_loss = 0
                running_size = 0        

                for data_ in loader:
                    x, y = data_[0].to(self.device), data_[1].to(self.device)

                    if phase == "train":
                        optimizer.zero_grad()
                        context = self.active_context
                        self.context_error[context] = torch.cat(
                            [self.context_error[context], self.context_error_0], dim=0
                        )

                        # Get prediction with current context. 
                        W, y_pred = self.predict_with_hnet(context, LSTM_, x, hx)

                       
                        # Compute loss using HUBER Loss
                        loss_task = F.huber_loss(y_pred, y, delta=delta)

                        # Calculate regularization loss if required
                        if kwargs.get('calc_reg', True) and self.active_context > 0:
                            loss_reg = self.calculate_regularization_loss(context, prev_hnet, kwargs)
                            loss_t = loss_task + beta * loss_reg
                        else:
                            loss_t = loss_task 

                        if regularizer is not None:
                            loss_t += regularizer(W, alpha, l1_ratio)

                        loss_t.backward()  # Backpropagation
                        optimizer.step()  # Update weights

                        modulation = loss_task.detach()  # Store modulation for analysis

                        # Update covariance for the current context
                        self.update_rolling_covariance(x.detach())

                        # Compute the current covariance
                        rolling_mean_covariance = self.compute_mean_covariance(self.rolling_covariances)

                        with torch.no_grad():
                            if self.confidence_context[self.active_context] > 0.9 and self.deviate_from_mean(modulation, self.active_context):
                                reactivation = False
                                self.new_context = True
                                if epoch == 0:
                                    for c in range(len(self.context_error)):
                                        self.thresholds_contexts[c] += 0.2      #  Final version 0.2      

                                # Covariance-based task detection
                                
                                similar_task_found, similar_task_index = self.is_similar_to_previous_tasks(
                                    rolling_mean_covariance 
                                )
                                if similar_task_found:
                                    print('Found similarities with other covariance matrix')
                                    self.active_context = similar_task_index
                                    print('New context is ', self.active_context)
                                    reactivation = True
                                    self.thresholds_contexts[self.active_context] =  2.2 # for stimulation data 1.01                 
                            
                                for context in range(len(self.context_error)):
                                    W = self.hnet(cond_id=context)
                                    model = RNN_Main_Model(
                                        num_features=self.model.num_features, 
                                        hnet_output=W,  
                                        hidden_size=self.model.hidden_size,
                                        num_layers=self.model.num_layers, 
                                        out_dims=self.model.out_features,  
                                        dropout=self.model.dropout_value, 
                                        LSTM_=LSTM_
                                    ).to(self.device)
                                    y_pred = model(x, hx)
                                    m = F.huber_loss(y_pred, y, delta=delta)
                                    print(m)
                                    thrs_context = self.thresholds_contexts[context]
                                    print(thrs_context * torch.mean(self.context_error[context][-1000:-1])) # Stimulation data [-100:-1]
                                    
                                    change_detect_epoch.append(epoch)
                                    prev_context.append(self.active_context)
                                    prev_min_loss.append(torch.min(self.context_error[self.active_context][-15:-1].min(), modulation).detach().cpu().numpy())
                                    prev_mean_loss.append(torch.mean(self.context_error[self.active_context][-1000:-1]).detach().cpu().numpy()) # Stimulation data [-100:-1]
                                    new_context.append(context)
                                    new_min_loss.append(m.detach().cpu().numpy())
                                    new_mean_loss.append(thrs_context * torch.mean(self.context_error[context][-1000:-1]).detach().cpu().numpy())  # Stimulation data [-100:-1]


                                    # Calculate similarity score
                                    diff = torch.abs(rolling_mean_covariance - self.task_covariances[context])
                                    similarity_score_ = diff.mean().item()
                                    print('Similarity', similarity_score_)
                                    similarity_scores.append(similarity_score_)


                                    if similar_task_found == False:
                                        #### Test using loss.
                                        if m < (thrs_context * torch.mean(self.context_error[context][-1000:-1])): # Stimulation data [-100:-1]
                                            reactivation = True
                                            self.active_context = context
                                            self.thresholds_contexts[context] = 2.2  # for stimulation data 1.01
                                            break

                                if not reactivation:
                                    self.confidence_context.append(0)
                                    self.active_context = len(self.context_error)
                                    self.n_contexts += 1
                                    self.context_error.append(self.context_error_0)
                                    prev_hnet = deepcopy(self.hnet)
                                    self.rolling_covariances = [] # Added on 08/08/2024
                            
                            else:
                                self.confidence_context[self.active_context] += (1 - self.confidence_context[self.active_context]) * 0.005
                                self.context_error[self.active_context][-1] = modulation
                                self.update_covariance_for_context(context)


                    else:
                        # Get prediction with current context. 
                        W, y_pred = self.predict_with_hnet(context, LSTM_, x, hx)

                        loss = F.huber_loss(y_pred, y, delta=delta)
                        loss_task = loss

                    assert torch.isfinite(loss_task)
                    running_loss += loss_task.item()
                    running_size += 1

                running_loss /= running_size
                if phase == "train":
                    train_losses.append(running_loss)
                else:
                    val_losses.append(running_loss)
                    if running_loss < best_loss:
                        best_loss = running_loss
                        best_model_wts = deepcopy(self.hnet.state_dict())
                        final_active_context = self.active_context
                        final_n_contexts = self.n_contexts
                        final_context_error = self.context_error
                        final_sim_scores = similarity_scores
                        not_increased = 0
                    else:
                        if epoch > 10:
                            not_increased += 1
                            if not_increased == early_stop:
                                for g in optimizer.param_groups:
                                    g['lr'] = g['lr'] / 2
                                not_increased = 0
                                end_train += 1
                            
                            if end_train == 1:
                                self.hnet.load_state_dict(best_model_wts)
                                self.context_error = final_context_error
                                self.active_context = final_active_context
                                self.n_contexts = final_n_contexts
                                similarity_scores = final_sim_scores if len(final_sim_scores) > 0 else ['No similarities']
                                print('Final active context :', self.active_context)
                                return self.hnet, np.array(train_losses), np.array(val_losses),\
                                     change_detect_epoch,prev_context, prev_min_loss,\
                                        prev_mean_loss, new_context, \
                                        new_min_loss,new_mean_loss, similarity_scores, self.active_context
            print('Num contexts after epoch ', epoch, len(self.context_error))                
            scheduler.step()

        self.hnet.load_state_dict(best_model_wts)
        self.context_error = final_context_error
        self.active_context = final_active_context
        self.n_contexts = final_n_contexts
        similarity_scores = final_sim_scores if len(final_sim_scores) > 0 else ['No similarities']
        print('Final active context :', self.active_context)
        
        return self.hnet, np.array(train_losses),  np.array(val_losses), change_detect_epoch,\
              prev_context, prev_min_loss,\
                  prev_mean_loss, new_context, \
                  new_min_loss,new_mean_loss, similarity_scores, self.active_context
                            

    def predict_with_hnet(self, context, LSTM_, x, hx):
        """
        Get predictions using the combination of hypernetwork and main model for the current context.
        
        Args:
            - context: the task or condition for which the hypernetwork is going to generate weights.
            - LSTM_ : whether the main model is a simple RNN or an LSTM
            - x : the batch input
            - hx : the initial hidden states for the model.
        """
        
        W = self.hnet(cond_id=context)
        model = RNN_Main_Model(
            num_features=self.model.num_features, 
            hnet_output=W,  
            hidden_size=self.model.hidden_size,
            num_layers=self.model.num_layers, 
            out_dims=self.model.out_features,  
            dropout=self.model.dropout_value, 
            LSTM_=LSTM_
        ).to(self.device)

        return W, model(x, hx)

    

    def initialize_hidden_states(self, batch_size, LSTM_):
        """Initialize hidden states for LSTM or other models."""
        if LSTM_:
            h0 = torch.randn(self.model.num_layers, batch_size, self.model.hidden_size, device=self.device) * 0.1
            c0 = torch.randn(self.model.num_layers, batch_size, self.model.hidden_size, device=self.device) * 0.1
            return (h0, c0)
        return torch.randn(self.model.num_layers, batch_size, self.model.hidden_size, device=self.device) * 0.1

    def calculate_regularization_loss(self, context, prev_hnet, kwargs):
        """Calculate regularization loss for the current context."""
        reg_targets = get_current_targets_NC(context, prev_hnet, len(self.context_error))
        prev_hnet_theta = None  # Update if needed
        prev_task_embs = None  # Update if needed
        loss_reg = calc_fix_target_reg_NC(
                                    self.hnet,
                                    context,
                                    len(self.context_error),
                                    targets=reg_targets,
                                    mnet=self.model,
                                    prev_theta=prev_hnet_theta,
                                    prev_task_embs=prev_task_embs,
                                )
        return loss_reg
