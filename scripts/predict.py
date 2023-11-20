# prediction.py
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the Python path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import get_model  # Adjust the import path as necessary
from data.dataloader import AudioDataset  # Adjust the import path as necessary
from torch.utils.data import DataLoader
import os

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_loader):
  ### What you call for model to perform inference?
    model.eval()

  ### List to store predicted phonemes of test data
    test_predictions = []
    ground_truth = []

  ### Which mode do you need to avoid gradients?
    with torch.inference_mode():

        for i, data in enumerate(tqdm(test_loader)):

            phoneme, groundtruth_AM = data
            ### Move data to device (ideally GPU)
            phoneme, groundtruth_AM = phoneme.to(device), groundtruth_AM.to(device)         
          
            predicted_AM = model(phoneme)
            predicted_AM.squeeze_()
            # print(predicted_AM.shape)
            # print(groundtruth_AM.shape)

          ### How do you store predicted_phonemes with test_predictions? Hint, look at eval 
            test_predictions.extend(predicted_AM.cpu().tolist())
            ground_truth.extend(groundtruth_AM.cpu())
    
    # print(len(test_predictions))
    return test_predictions, ground_truth

def main():
    config = {
        # Add configuration parameters if needed
    }

    # Load the trained model
    model = get_model('mnasnet1_0', weights=None).to(device)
    model.load_state_dict(torch.load('model_epoch_10.pth'))

    # Create the dataset and dataloader for testing
    dataset = AudioDataset(
        data_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/extract_phoneme_processed'),
        am_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/AMs_final.csv'),
        gender='female',
        phoneme_idx=4,
        am_idx=13,
        MAX_LEN=32,
        partition="test"  # Use test partition
    )
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Run the test
    predictions, ground_truth = test(model, test_loader)
    # Print some sample predictions
    print("Sample Predictions:")
    for i in range(min(10, len(predictions))):  # Print the first 10 predictions
        print(f"Prediction: {predictions[i]}, Ground Truth: {ground_truth[i]}")

    # Optionally, save the results to a file
    # with open('predictions.csv', 'w') as f:
    #     for pred, gt in zip(predictions, ground_truth):
    #         f.write(f"{pred},{gt}\n")
    # Handle the predictions as needed (e.g., save to file, print, etc.)

if __name__ == '__main__':
    main()