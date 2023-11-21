```
.
│
├── README.md                  # The main documentation file for the project
├── run_pipeline.py            # The main script to run the entire pipeline of the project. 
│
├── data/                      # Directory for data-related utilities
│   └── README_data.md         # Description of dataset
│
├── models/                    # Directory for model architecture definitions
│   └── model.py               # Contains the definition of the model used in the project.
│
├── scripts/                   # Directory for executable scripts
│   ├── train.py               # Script to run training
│   ├── evaluate.py            # Script to run evaluation
│   ├── predict.py             # Contains the prediction function for the model.
│   ├── Extract_AM_points.py   # Extracting anthropometric measurements (AM) points from the face data.
│   └── identify.py            # Contains the main function for identifying phonemes and facial features.
│
├── configs/                   # Configuration files for model, training, etc.
│   └── model_config.yaml      # YAML configuration file for model hyperparameters
│
├── utils/                     # Directory for utility code
│   ├── dataloader.py          # Defines the AudioDataset class for loading and preprocessing the audio data.
│   └── dataloader_test.py     # Contains tests for the AudioDataset class
│
├── requirements.txt           # Python dependencies required for the project
└── .gitignore                 # Specifies untracked files to ignore


```
