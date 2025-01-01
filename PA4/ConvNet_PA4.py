import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)    # First convolutional layer: input channels = 3 (RGB), output channels = 32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # Second convolutional layer: input channels = 32, output channels = 64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Third convolutional layer: input channels = 64, output channels = 128
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # Fourth convolutional layer: input channels = 128, output channels = 256
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) # Fifth convolutional layer: input channels = 256, output channels = 512
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                   # Max pooling layer to reduce the spatial dimensions by half
        self.fc11 = nn.Linear(in_features=128 * 4 * 4, out_features=256)                     # Fully connected layer after convolution layers: input features = 128 * 4 * 4 (flattened), output features = 256
        self.fc21 = nn.Linear(in_features=256 * 2 * 2, out_features=256)                     # Fully connected layer after convolution layers: input features = 256 * 2 * 2 (flattened), output features = 256
        self.fc31 = nn.Linear(in_features=512 * 1 * 1, out_features=256)                      # Fully connected layer after convolution layers: input features = 512 * 1 * 1 (flattened), output features = 256
        self.fc2 = nn.Linear(in_features=256, out_features=10)                              # Fully connected layer: input features = 256, output features = 10 (for 10 classes)
        self.fc41 = nn.Linear(in_features=512 * 1 * 1, out_features=256)                    # Fully connected layer after convolution layers: input features = 512 * 1 * 1 (flattened), output features = 256
        self.fc42 = nn.Linear(in_features=256, out_features=128)                            # Fully connected layer: input features = 256, output features = 128
        self.fc43 = nn.Linear(in_features=128, out_features=10)                             # Fully connected layer: input features = 128, output features = 10 (for 10 classes)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-3")
            exit(0)
        
    def model_1(self, x):
        ################################################
        ### 3 convolutional layers + 2 linear layers ###
        ################################################
        x = self.pool(torch.relu(self.conv1(x)))   # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))   # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv3(x)))   # Apply third convolutional layer followed by ReLU activation and pooling
        x = x.view(-1, 128 * 4 * 4)                # Flatten the tensor
        x = torch.relu(self.fc11(x))                # Apply first fully connected layer followed by ReLU activation
        x = self.fc2(x)                            # Apply the final fully connected layer
        return x            

    def model_2(self, x):
        ################################################
        ### 4 convolutional layers + 2 linear layers ###
        ################################################
        x = self.pool(torch.relu(self.conv1(x)))   # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))   # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv3(x)))   # Apply third convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv4(x)))   # Apply fourth convolutional layer followed by ReLU activation and pooling
        x = x.view(-1, 256 * 2 * 2)                # Flatten the tensor
        x = torch.relu(self.fc21(x))                # Apply first fully connected layer followed by ReLU activation
        x = self.fc2(x)                            # Apply the final fully connected layer
        return x   

    def model_3(self, x):
        ################################################
        ### 5 convolutional layers + 2 linear layers ###
        ################################################
        x = self.pool(torch.relu(self.conv1(x)))   # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))   # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv3(x)))   # Apply third convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv4(x)))   # Apply fourth convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv5(x)))   # Apply fifth convolutional layer followed by ReLU activation and pooling
        x = x.view(-1, 512 * 1 * 1)                # Flatten the tensor
        x = torch.relu(self.fc31(x))                # Apply first fully connected layer followed by ReLU activation
        x = self.fc2(x)                            # Apply the final fully connected layer
        return x
    
    def model_4(self, x):
        ################################################
        ### 5 convolutional layers + 3 linear layers ###
        ################################################
        x = self.pool(torch.relu(self.conv1(x)))   # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))   # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv3(x)))   # Apply third convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv4(x)))   # Apply fourth convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv5(x)))   # Apply fifth convolutional layer followed by ReLU activation and pooling
        x = x.view(-1, 512 * 1 * 1)                # Flatten the tensor
        x = torch.relu(self.fc41(x))               # Apply first fully connected layer followed by ReLU activation
        x = torch.relu(self.fc42(x))               # Apply second fully connected layer followed by ReLU activation
        x = self.fc43(x)                           # Apply the final fully connected layer
        return x    
