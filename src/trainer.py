import torch
import numpy as np
import torch
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

from src.helpers import * 
from src.sequence_datasets import *


device = torch.device('cuda:0') #suposed to be cuda
#device = torch.device('cpu') #suposed to be cuda
dtype = torch.float32


# Function to calculate loss (including regularization if applicable)
def compute_loss(model, X_batch, y_batch, delta, regularizer=None, l1_ratio=0.5, alpha=1e-5, layer_type='rnn'):
    output = torch.squeeze(model(X_batch))
    loss = huber_loss(output, y_batch, delta=delta)

    # Apply regularization if provided
    if regularizer is not None:
        loss += regularizer(model, l1_ratio, alpha, layer_type)
    
    return loss

# Training function with early stopping and regularization support
def train_model(model, X, Y, X_val, Y_val,
                lr=0.0001, lr_step_size=10, lr_gamma=0.9,
                sequence_length_LSTM=10, batch_size_train=3, batch_size_val=3,
                num_epochs=1000, delta=8, regularizer=None, 
                layer_type='rnn', l1_ratio=0.5, alpha=1e-5, 
                early_stop=5):
    
    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Prepare datasets and data loaders
    train_dataset = SequenceDataset(Y, X, sequence_length=sequence_length_LSTM)
    val_dataset = SequenceDataset(Y_val, X_val, sequence_length=sequence_length_LSTM)
    
    loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    
    # Initialize tracking for best model and losses
    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    
    train_losses, val_losses = [], []
    not_improved_count = 0
    early_stop_counter = 0

    torch.autograd.set_detect_anomaly(True)  # For debugging
    
    # Training loop over epochs
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()
            loader = loader_train if is_train else loader_val

            running_loss = 0.0
            running_size = 0
            
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    loss = compute_loss(model, X_batch, y_batch, delta, regularizer, l1_ratio, alpha, layer_type)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item()
                running_size += 1

            avg_loss = running_loss / running_size
            if is_train:
                train_losses.append(avg_loss)
            else:
                val_losses.append(avg_loss)

                # Early stopping logic
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_wts = deepcopy(model.state_dict())
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                    if not_improved_count >= early_stop:
                        # Reduce learning rate and track if early stopping threshold is met
                        print(f"Learning rate halved after {early_stop} non-improving epochs")
                        for g in optimizer.param_groups:
                            g['lr'] /= 2
                        early_stop_counter += 1
                        not_improved_count = 0

                    if early_stop_counter >= 2:
                        print(f"Early stopping at epoch {epoch}, best epoch was {epoch - not_improved_count}")
                        model.load_state_dict(best_model_wts)
                        return np.array(train_losses), np.array(val_losses)

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch:03d} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Load the best model's weights before returning
    model.load_state_dict(best_model_wts)
    print(f"Training completed. Best epoch: {epoch - not_improved_count}")
    
    return np.array(train_losses), np.array(val_losses)