import torch
import torch.nn as nn

# Defining Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, mode):
        super(Autoencoder, self).__init__()
        ##########################
        ### Linear AutoEncoder ###
        ##########################
        # Encoder layers with Linear transformation
        self.linearEncoder = nn.Sequential(
            nn.Linear(28 * 28, 256),  # Input layer to 256 neurons
            nn.ReLU(),
            nn.Linear(256, 128),      # Second layer with 128 neurons
            nn.ReLU()
        )
        # Decoder layers with Linear transformation
        self.linearDecoder = nn.Sequential(
            nn.Linear(128, 256),      # First Layer with 256 neurons
            nn.ReLU(),
            nn.Linear(256, 28 * 28),  # Output layer with 784 neurons (28*28)
            nn.Sigmoid()              # Output layer with values in range [0, 1]
        )
        
        #################################
        ### Convolutional AutoEncoder ###
        #################################

        # Encoder layers with Convolutional transformation
        self.convEncoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # Convolutional layer (input channels: 1, output channels: 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # Max pooling layer to reduce spatial dimensions by half
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Convolutional layer (input channels: 16, output channels: 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)         # Max pooling layer to reduce spatial dimensions by half
        )
        # Decoder layers with Convolutional transformation
        self.convDecoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Convolutional layer (input channels: 32, output channels: 16)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling layer to double the spatial dimensions
            nn.Conv2d(16, 8, kernel_size=3, padding=1),   # Convolutional layer (input channels: 16, output channels: 8)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling layer to double the spatial dimensions
            nn.Conv2d(8, 1, kernel_size=3, padding=1),    # Final convolutional layer to output 1 channel
            nn.Sigmoid()  # Use sigmoid to bring pixel values to [0, 1]
        )

        ######################
        ### mode selection ###
        ######################
        self.mode = mode

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else:
            print("Invalid mode ", mode, "selected. Please select mode 1 or 2.")
            exit(0)

    ##################################################################################
    ### Model 1 with two linear layers in Encoder and two linear layers in Decoder ###
    ##################################################################################
    def model_1(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        # Encoder forward pass
        x = self.linearEncoder(x)
        # Decoder forward pass
        x = self.linearDecoder(x)
        # Reshape the output back to image size
        x = x.view(-1, 1, 28, 28)
        return x
    
    ##################################################################################################
    ### Model 2 with two convolutional layers in Encoder and three convolutional layers in Decoder ###
    ##################################################################################################
    def model_2(self, x):
        # Encoder forward pass
        x = self.convEncoder(x)
        # Decoder forward pass
        x = self.convDecoder(x)
        return x

