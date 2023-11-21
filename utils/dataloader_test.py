from torch.utils.data import DataLoader
from dataloader import AudioDataset  # make sure dataloader.py is in the same directory or appropriately in your PYTHONPATH
import os

def test_dataloader():
    # Parameters for your dataset
    # data_path = '~/Documents/Workspace/11785\ project/penstate_data/extract_phoneme_processed'
    # am_path = '~/Documents/Workspace/11785\ project/penstate_data/AMs_final.csv'
    data_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/extract_phoneme_processed')
    am_path = os.path.expanduser('~/Documents/Workspace/11785 project/penstate_data/AMs_final.csv')
    gender = 'female'
    phoneme_idx = 4
    am_idx = 13
    MAX_LEN = 32
    partition = "train"

    # Create an instance of your dataset
    dataset = AudioDataset(data_path, am_path, gender, phoneme_idx, am_idx, MAX_LEN, partition)

    # Wrap your dataset in a DataLoader
    dataloader = DataLoader(dataset, num_workers=0, batch_size=64, shuffle=True)

    # Iterate over the DataLoader
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets: {targets}")
        # Here you could also add the code to send inputs to your model, etc.
        # but for simple testing, printing the shapes and targets is often enough.
        if i >= 10:  # Print only the first 10 batches to avoid a long loop during testing
            break

if __name__ == '__main__':
    test_dataloader()
