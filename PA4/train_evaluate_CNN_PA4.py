from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard.writer import SummaryWriter
from ConvNet_Sina import ConvNet 
import argparse
import numpy as np
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, writer):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}')):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # Count how many predictions are correct by comparing with ground truth 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f} | Accuracy: {}/{} ({:.2f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    
    # Log training loss and accuracy to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    
    return train_loss, train_acc
    


def test(model, device, test_loader, criterion, epoch, writer):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader, desc='Testing')):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Count how many predictions are correct by comparing with ground truth
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f} | Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    # Log test loss and accuracy to TensorBoard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    return test_loss, accuracy
    

def run_main(FLAGS):
    '''
    For Nvidia GPUs, use the below code for using CUDA
    '''
    #Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    #Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected:", device)

    ''' 
    Since I have a Mac with M2 chip, I changed this part of code.
    For Apple GPUs, use the below code for using Metal Performance Shaders (MPS)
    '''
    # use_mps = torch.backends.mps.is_available()
    # device = torch.device("mps" if use_mps else "cpu")
    # print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer function
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.CIFAR10('./data/', train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10('./data/', train=False, download=True, transform=transform)

    # Create dataloaders for training and testing
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, shuffle=False, num_workers=2)
    
    # Set up TensorBoard writer to log training and evaluation metrics
    writer = SummaryWriter(FLAGS.log_dir)

    best_accuracy = 0.0

    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"Epoch: {epoch}\n")
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size, writer)
        test_loss, test_accuracy = test(model, device, test_loader, criterion, epoch, writer)
        print("-"*100 + "\n")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
    # Close the TensorBoard writer
    writer.close()
    print("accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)