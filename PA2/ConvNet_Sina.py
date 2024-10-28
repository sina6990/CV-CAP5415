import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        # Definig required layers for Model 1
        self.fc1_m1 = nn.Linear(28 * 28, 100) # Input size: 28x28 (flattened image), Output: 100 neurons
        self.fc2_m1 = nn.Linear(100, 10)      # Input: 100 neurons, Output: 10 classes (digits 0-9)

        # Defining required layers for Model 2 & 3
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)  # Convolution layer: 1 input channel, 40 output channels, 5x5 filter
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       # Max pooling layer: 2x2 filter with stride 2
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1) # Convolution layer: 40 input channels, 40 output channels, 5x5 filter 
        self.fc1_m2_3 = nn.Linear(40 * 4 * 4, 100)              # Flattened feature map to 100 neurons
        self.fc2_m2_3 = nn.Linear(100, 10)                      # Output: 10 classes (digits 0-9)

        # Defining required layers for Model 4
        self.fc1_m4 = nn.Linear(40 * 4 * 4, 100)  # First fully connected layer with 100 neurons
        self.fc2_m4 = nn.Linear(100, 100)         # Second fully connected layer with 100 neurons
        self.fc3_m4 = nn.Linear(100, 10)

        # Defining required layers for Model 5
        self.fc1_m5 = nn.Linear(40 * 4 * 4, 1000) # First fully connected layer with 1000 neurons
        self.fc2_m5 = nn.Linear(1000, 1000)       # Second fully connected layer with 1000 neurons
        self.fc3_m5 = nn.Linear(1000, 10)         # Output layer: 10 classes (digits 0-9)
        self.dropout = nn.Dropout(0.5)            # Dropout layer with 50% probability


        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        #################################
        ### One fully connected layer ###
        #################################
        X = torch.flatten(X, start_dim=1) # Flatten the input tensor
        X = self.fc1_m1(X)                # Apply first fully connected layer                          
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.fc2_m1(X)                # Apply second fully connected layer
        return X

    # Use two convolutional layers.
    def model_2(self, X):
        #############################################################
        ### Two convolutional layers + one fully connnected layer ###
        #############################################################
        X = self.conv1(X)                 # Apply first convolutional layer
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.pool(X)                  # Apply max pooling
        X = self.conv2(X)                 # Apply second convolutional layer
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.pool(X)                  # Apply max pooling
        X = torch.flatten(X, start_dim=1) # Flatten the input tensor
        X = self.fc1_m2_3(X)              # Apply first fully connected layer
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.fc2_m2_3(X)              # Apply second fully connected layer
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        #######################################################################
        ### Two convolutional layers + one fully connected layer, with ReLU ###
        #######################################################################
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = F.relu(X)
        X = self.pool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.fc1_m2_3(X)
        X = F.relu(X)
        X = self.fc2_m2_3(X)
        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        ########################################################################
        ### Two convolutional layers + two fully connected layers, with ReLU ###
        ########################################################################
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = F.relu(X)
        X = self.pool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.fc1_m4(X)
        X = F.relu(X)
        X = self.fc2_m4(X)
        X = F.relu(X)
        X = self.fc3_m4(X)
        return X

    # Use Dropout now.
    def model_5(self, X):
        ##################################################################################
        ### Two convolutional layers + two fully connected layers, with ReLU + Dropout ###
        ##################################################################################
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = F.relu(X)
        X = self.pool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.fc1_m5(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.fc2_m5(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.fc3_m5(X)
        return X
    
