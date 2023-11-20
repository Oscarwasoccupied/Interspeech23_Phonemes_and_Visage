import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path

# Add the project directory to the Python path for relative imports
sys.path.append(str(Path(__file__).resolve().parent))

from data.dataloader import AudioDataset
from models.model import get_model
from scripts.train import train
from scripts.evaluate import evaluate
from scripts.predict import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_pipeline():
    # Setup paths and parameters
    default_root_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/extract_phoneme_processed')
    am_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/AMs_final.csv')
    gender = "female"

    # am_list = [89, 13, 88, 51, 14, 52, 58, 17, 59, 73, 71, 22, 15, 69, 70, 96, 42, 7, 37, 31]
    # phoneme_list = [37]
    am_list = [88]
    phoneme_list = [4]
    g_flag = "F" if gender == "female" else "M"

    for k in phoneme_list:
        for j in am_list:
            # if k == 4 and j in [89, 13, 88, 51, 14]:
            #     continue

            phoneme_idx = k
            am_idx = j
            folder_name = f"estimations/{g_flag}_phoneme{phoneme_idx}_AM{am_idx}"
            os.makedirs(folder_name, exist_ok=True)  # Create the directory if it doesn't exist

            for x in range(2):
                # Dataset and DataLoader setup
                config = {'epochs': 10, 'learning_rate': 0.001, 'batch_size': 32}
                train_data = AudioDataset(default_root_path, am_path, gender, phoneme_idx, am_idx, MAX_LEN=32, partition="train")
                train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
                val_data = AudioDataset(default_root_path, am_path, gender, phoneme_idx, am_idx, MAX_LEN=32, partition="val1")
                val_loader = DataLoader(val_data, batch_size=config['batch_size'])
                test_data = AudioDataset(default_root_path, am_path, gender, phoneme_idx, am_idx, MAX_LEN=32, partition="val1")
                test_loader = DataLoader(test_data, batch_size=config['batch_size'])

                # Calculate the mean of target_am
                all_am = None
                for i, data in enumerate(train_loader):
                    _, target_am = data
                    target_am = target_am.to(device) 
                    all_am = target_am if all_am is None else torch.cat([all_am, target_am])

                with open(f"{folder_name}/{folder_name.split('/')[-1]}_times{x}.txt", "a+") as f:
                    f.write(f'{phoneme_idx},{am_idx},{all_am.mean().item()}\n')

                # Train, Evaluate and Predict
                model = get_model('mnasnet1_0', weights=None)
                model = model.to(device)  # Move model to GPU
                # phoneme, AM = next(iter(train_loader))
                criterion = torch.nn.MSELoss() #Defining Loss function 
                optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) #Defining Optimizer
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * config['epochs']))
                
                torch.cuda.empty_cache()
                # gc.collect()
                
                best_mse = 1.0 ### Monitor best accuracy in your run

                for epoch in range(config['epochs']):
                    print("\nEpoch {}/{}".format(epoch+1, config['epochs']))

                    train_loss = train(model, optimizer, criterion, train_loader, scheduler)
                    MSE = evaluate(model, val_loader)

                    print("\tTrain Loss: ", train_loss)
                    print("\tValidation MSE: ", MSE)

                    ### Save checkpoint if accuracy is better than your current best
                    if MSE < best_mse:
                        best_mse = MSE
                    ### Save checkpoint with information you want
                        torch.save({'epoch': epoch,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'loss': train_loss,
                              'learning rate': scheduler.get_last_lr()[0],
                              'mse': MSE}, 
                        f"{folder_name}/{folder_name.split('/')[-1]}_model_checkpoint_times{x}.pth")

                # predict the result 
                predictions, ground_truth = test(model, test_loader)

                # Save predictions to a CSV file
                with open(f"{folder_name}/{folder_name.split('/')[-1]}_predictions_times{x}.csv", "w+") as f:
                    f.write("person, label, prediction\n")
                    for i in range(len(predictions)):
                        f.write(f"{i},{ground_truth[i]},{predictions[i]}\n")

if __name__ == '__main__':
    run_pipeline()
