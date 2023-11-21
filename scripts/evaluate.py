# evaluate.py
import torch
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Again, add the parent directory to the Python path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import get_model  # Adjust the import path as necessary
from utils.dataloader import AudioDataset   # Adjust the import path as necessary
from torch.utils.data import DataLoader
import os

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader):

    model.eval() # set model in evaluation mode

    AM_true_list = []
    AM_pred_list = []

    for i, data in enumerate(dataloader):

        phoneme, AM = data
        ### Move data to device (ideally GPU)
        phoneme, AM = phoneme.to(device), AM.to(device) 

        with torch.inference_mode(): # makes sure that there are no gradients computed as we are not training the model now
            ### Forward Propagation
            ### Get Predictions
            predicted_AM = model(phoneme)
            # print(predicted_AM)
        
        ### Store Pred and True Labels
        AM_pred_list.extend(predicted_AM.cpu().tolist())
        AM_true_list.extend(AM.cpu().tolist())
        
        # Do you think we need loss.backward() and optimizer.step() here?
    
        del phoneme, AM, predicted_AM
        torch.cuda.empty_cache()

    ###############################################################################################
    # print(AM_pred_list[1000:3100])
    # print(AM_true_list)
    # print(len(AM_pred_list))
    # print(len(AM_true_list))
    ###############################################################################################
    
    # print("Number of equals between two list: ", sum(a == b for a,b in zip(AM_pred_list, AM_true_list)))
    
    ### Calculate Accuracy
    MSE = mean_squared_error(AM_pred_list, AM_true_list)
    r2_score_acc = r2_score(AM_pred_list, AM_true_list)
    MAE = mean_absolute_error(AM_pred_list, AM_true_list)
    print(f"Evaluation - MSE: {MSE}, R2: {r2_score_acc}, MAE: {MAE}")
    
    return MSE

def main():
    config = {
        # Add configuration parameters if needed
    }

    # Load the trained model
    model = get_model('mnasnet1_0', weights=None).to(device)
    model.load_state_dict(torch.load('model_epoch_10.pth'))

    # Create the dataset and dataloader for evaluation
    dataset = AudioDataset(
        data_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/extract_phoneme_processed'),
        am_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/AMs_final.csv'),
        gender='female',
        phoneme_idx=4,
        am_idx=13,
        MAX_LEN=32,
        partition="val1"  # Use validation partition
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # No need to shuffle for evaluation

    # Run the evaluation
    eval(model, dataloader)

if __name__ == '__main__':
    main()