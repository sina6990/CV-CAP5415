import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Autoencoder_model import Autoencoder
from tqdm import tqdm
from tensorboard import notebook
import matplotlib.pyplot as plt

# Define Training Function
def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()    # Set the model to training mode
    running_loss = 0
    total_samples = 0
    for images, _ in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
        images = images.to(device)   # Move images to the GPU if available
        batch_size = images.size(0)  # Get the batch size
        total_samples += batch_size  # Accumulate total samples processed
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, images)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Update running loss
        running_loss += loss.item() * batch_size
    
    # Calculate average loss
    avg_loss = running_loss / total_samples
    # Log the training loss
    writer.add_scalar('Training Loss', avg_loss, epoch)
    print(f'Epoch {epoch}, Training Loss: {avg_loss:.2f}')

# Define Testing Function
def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()      # Set the model to evaluation mode 
    running_loss = 0
    total_samples = 0

    # Store original and reconstructed images for each digit class (0-9)
    class_samples = {i: {'orig': [], 'recon': []} for i in range(10)}

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Testing Epoch {epoch}'):
            images = images.to(device)   # Move images to the GPU if available
            batch_size = images.size(0)  # Get the batch size
            total_samples += batch_size  # Accumulate total samples processed
            # Forward pass to get reconstructed images
            outputs = model(images)
            # Compute loss
            loss = criterion(outputs, images)
            running_loss += loss.item() * batch_size
            # Store samples for visualization
            for idx in range(len(labels)):
                label = labels[idx].item()
                if len(class_samples[label]['orig']) < 2:  # Store only 2 samples per class
                    class_samples[label]['orig'].append(images[idx].cpu())
                    class_samples[label]['recon'].append(outputs[idx].cpu())
        # Calculate average loss
        avg_loss = running_loss / total_samples
        # Log the testing loss
        writer.add_scalar('Testing Loss', avg_loss, epoch)
        print(f'Epoch {epoch}, Testing Loss: {avg_loss:.2f}')

        # Create directory based on model mode
        model_dir = f'Model {model.mode} images' 
        os.makedirs(model_dir, exist_ok=True)

        # Visualization of original and reconstructed images
        fig, axes = plt.subplots(4, 10, figsize=(20, 8))
        plt.suptitle(f'Model {model.mode} - Original vs Reconstructed Images (Epoch {epoch})', y=1.02)

        # Add column labels for digits 0-9
        for j in range(10):
            axes[0, j].set_title(f'Digit {j}')

        # Add row labels
        rows = ['Original 1', 'Reconstructed 1', 'Original 2', 'Reconstructed 2']
        for i in range(4):
            axes[i, 0].set_ylabel(rows[i], rotation=90, size = 'large')
        
        # Plot images
        for digit in range(10):
            if len(class_samples[digit]['orig']) >= 2:
                # First sample
                axes[0, digit].imshow(class_samples[digit]['orig'][0].view(28, 28), cmap='gray')
                axes[1, digit].imshow(class_samples[digit]['recon'][0].view(28, 28), cmap='gray')
                # Second sample
                axes[2, digit].imshow(class_samples[digit]['orig'][1].view(28, 28), cmap='gray')
                axes[3, digit].imshow(class_samples[digit]['recon'][1].view(28, 28), cmap='gray')
            
            # Turn off axis for all subplots
            for i in range(4):
                axes[i, digit].set_xticks([])
                axes[i, digit].set_yticks([])
        
        plt.tight_layout()
        save_path = os.path.join(model_dir, f'reconstruction_epoch_{epoch}.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def run_main(FLAGS):
    # Device agnostic code to use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Model, Define Loss Function and Optimizer
    model = Autoencoder(FLAGS.mode).to(device)

    # Mean Squared Error Loss Function
    criterion = nn.MSELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    # Load MNIST dataset with transform to normalize the images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    writer = SummaryWriter(log_dir=FLAGS.log_dir)

    for epoch in range(1, FLAGS.num_epochs+1):
        # Step 7: Train the Model
        train(model, device, train_loader, criterion, optimizer, epoch, writer)
        test(model, device, test_loader, criterion, epoch, writer)
        print('---------------------------------------------------------\n')

    # Compare Number of Parameters in Encoder and Decoder
    if model.mode == 1:
        encoder_params = sum(p.numel() for p in model.linearEncoder.parameters())
        decoder_params = sum(p.numel() for p in model.linearDecoder.parameters())
        print(f'Number of parameters in Encoder: {encoder_params}')
        print(f'Number of parameters in Decoder: {decoder_params}')

        total_params = encoder_params + decoder_params
        print(f'Total number of parameters in Linear Autoencoder: {total_params}')
    else:
        encoder_params = sum(p.numel() for p in model.convEncoder.parameters())
        decoder_params = sum(p.numel() for p in model.convDecoder.parameters())
        print(f'Number of parameters in Encoder: {encoder_params}')
        print(f'Number of parameters in Decoder: {decoder_params}')

        total_params = encoder_params + decoder_params
        print(f'Total number of parameters in Convolutional Autoencoder: {total_params}')

    writer.close()

if __name__ == '__main__':
    # Set parameters for Argument Parser
    parser = argparse.ArgumentParser("Autoencoder Exercise")
    parser.add_argument('--mode', type=int, default=1, help='Select mode 1 or 2')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for DataLoader')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for saving logs')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)