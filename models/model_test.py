import torch
from models import model

# Testing models/model.py module
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.get_model('mnasnet1_0', weights=None)
model.to(device)
print(model)  # Check the modified layers to ensure they are as expected
