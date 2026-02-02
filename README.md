# Pseudo Labeling on MIT300/SALICON Dataset

A Python project for generating pseudo labels on saliency datasets using deep learning models.

## Features

- **Dataset Support**: MIT300 and SALICON saliency datasets
- **Model Inference**: Run pre-trained saliency models on images
- **Pseudo Label Generation**: Automatically generate annotation labels
- **Data Processing**: Utilities for preprocessing and augmentation

## Installation

1. Clone or create the project in your workspace
2. Install Python 3.8 or higher
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading utilities
│   ├── models.py            # Model definitions and inference
│   └── pseudo_labeling.py   # Pseudo labeling pipeline
├── data/                    # Dataset storage
├── models/                  # Model weights
├── notebooks/               # Jupyter notebooks for exploration
├── requirements.txt
└── README.md
```

## Usage

### Basic Setup
```python
from src.data_loader import MITDataset, SALICONDataset
from src.models import SaliencyModel
from src.pseudo_labeling import generate_pseudo_labels

# Load dataset
dataset = MITDataset('path/to/mit300')
# or
dataset = SALICONDataset('path/to/salicon')

# Initialize model
model = SaliencyModel(pretrained=True)

# Generate pseudo labels
labels = generate_pseudo_labels(dataset, model)
```

## Next Steps

1. Prepare your dataset (MIT300 or SALICON)
2. Download or train saliency models
3. Run the pseudo labeling pipeline
4. Use generated labels for downstream tasks

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, OpenCV
- Jupyter (optional, for notebooks)

See `requirements.txt` for full dependency list.

## License

MIT
