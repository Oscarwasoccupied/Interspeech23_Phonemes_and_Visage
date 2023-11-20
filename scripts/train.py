import torch
import sys
from pathlib import Path

# Add the parent directory of 'scripts' to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# - Path(__file__): This gets the path of the current Python script.
# - .resolve(): This resolves any symbolic links in the path.
# - .parent.parent: This gets the parent directory of the current script's directory (i.e., it goes up two levels).
# - str(): This converts the Path object to a string.
# - sys.path.append(): This adds the string to the Python path.
from models.model import get_model  # Now Python knows where to find models

# from ..models.model import get_model # Replace with your actual model import
from data.dataloader import AudioDataset  # Replace with your actual dataset import
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import os

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, criterion, dataloader, scheduler):

    model.train()
    train_loss = 0.0 #Monitoring Loss
    
    #########################################################
    # AM_true_list = []
    # AM_pred_list = []
    #########################################################
    
    for iter, (phoneme, AM) in enumerate(dataloader):
        # print(iter, data)
        ### Move Data to Device (Ideally GPU)
        phoneme = phoneme.to(device)
        AM = AM.to(device)
        # print(AM.shape)

        ### Forward Propagation
        preds_AM = model(phoneme)

        ### Loss Calculation
        
        preds_AM = torch.squeeze(preds_AM)
        # print(preds_AM)
        # print(preds_AM.shape)model = models.shufflenet_v2_x1_0(weights=None).to(device)
        loss = criterion(preds_AM, AM)
        train_loss += loss.item()
        
        #########################################################
        ### Store Pred and True Labels
        # AM_pred_list.extend(preds_AM.cpu().tolist())
        # AM_true_list.extend(AM.cpu().tolist())
        #########################################################

        ### Initialize Gradients
        optimizer.zero_grad()

        ### Backward Propagation
        loss.backward()

        ### Gradient Descent
        optimizer.step()

        ### Update the learning rate
        scheduler.step()  # This should come after optimizer.step()

        # if iter % 20 == 0:
        #     print("iter =", iter, "loss =",loss.item())
    train_loss /= len(dataloader)
    print("Learning rate = ", scheduler.get_last_lr()[0])
    print("Train loss = ", train_loss)
    
    #########################################################
    # print(AM_pred_list)
    # print(AM_true_list)
    # print(len(AM_pred_list))
    # print(len(AM_true_list))
    # accuracy = mean_squared_error(AM_pred_list, AM_true_list)
    # print("Train MSE accuracy: ", accuracy)
    #########################################################
    
    # scheduler.step() # add schedule learning rate
    return train_loss


def main():
    # Load configuration and initialize everything
    # For example, you might read a config file or command line arguments
    config = {
        'epochs': 10,  # Replace with your actual epochs
        'learning_rate': 0.001,  # Replace with your actual learning rate
        # Add other configuration parameters if needed
    }

    # Create the model
    model = get_model('mnasnet1_0', weights=None).to(device)

    # Create the dataset and dataloader
    dataset = AudioDataset(
        data_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/extract_phoneme_processed'),
        am_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/AMs_final.csv'),
        gender = 'female',
        phoneme_idx = 4,
        am_idx = 13,
        MAX_LEN = 32,
        partition = "train"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Optionally define a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    for epoch in range(config['epochs']):
        train_loss = train(model, optimizer, criterion, dataloader, scheduler)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss}")

        # Save the model checkpoints
        # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()