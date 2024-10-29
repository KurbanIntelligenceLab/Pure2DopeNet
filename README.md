

# Multimodal Neural Network-Based Predictive Modeling of Nanoparticle Properties from Pure Compounds

This is the official repository for `Multimodal Neural Network-Based Predictive Modeling of Nanoparticle Properties from Pure Compounds
` by Can Polat, Mustafa Kurban, and Hasan Kurban. This repository is maintained by Can Polat ([CalciumNitrade](https://github.com/CalciumNitrade)) and 
contains scripts for training and testing the `Pure2DopeNet` model as well as its model script. 
## Requirements

To run these scripts, ensure that the following Python packages are installed:

- `torch`
- `torchvision`
- `numpy`
- `tqdm`
- `argparse`

To install all required dependencies, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Data

You can download the data for compounds from the following link: [Pure2DopeNet Data](https://tamucs-my.sharepoint.com/:f:/g/personal/hasan_kurban_tamu_edu/EkHuqTEFjipEjMuIN8Ll4ssBKFMf_YipO87vFw97_kkj9A?e=uS2OTf)

## Usage

### Training

#### Command-Line Arguments for `train.py`

The training script accepts several command-line arguments to customize training:

- `--num_epochs`: Number of epochs for training (default: 50).
- `--learning_rate`: Learning rate for the optimizer (default: 0.001).
- `--batch_size`: Batch size for training (default: 64).
- `--root`: Root directory for saving training results.
- `--data_folder`: Folder containing the dataset (images and labels).
- `--csv_file`: CSV file containing dataset labels.
- `--output_dir`: Directory for output results.
- `--seed`: Random seed for reproducibility.
- `--physical`: Physical property to predict (e.g., `normalized_homo`).
- `--vector_type`: Text vector type corresponding to the physical property.

#### Example Command for Training

```bash
python train.py --num_epochs 100 --learning_rate 0.0005 --batch_size 32 --root /path/to/root \
--data_folder /path/to/data_folder --csv_file /path/to/csv_file.csv --output_dir /path/to/output \
--seed 42 --physical normalized_homo --vector_type text_vector_homo
```

#### Training Process

The training script performs the following steps:

1. **Data Loading**: Loads training and validation datasets using `MultimodalDataloader`.
2. **Training Loop**: Trains the model over the specified number of epochs, optimizing it using Mean Squared Error (MSE) loss and the Adam optimizer.
3. **Validation**: Validates the model on each epoch and saves the model's state when it achieves the lowest validation loss.
4. **Logging and Saving**: Logs training and validation losses at each epoch, saving them to text files. It also saves the final model's parameters and training duration.

### Testing

The testing script (`test.py`) is designed to evaluate a trained model using a specific test dataset.

#### Command-Line Arguments for `test.py`

- `--data_folder`: Folder containing the test dataset.
- `--csv_file`: CSV file containing test dataset labels.
- `--batch_size`: Batch size for testing (default: 1000).
- `--physical`: Physical property to predict (e.g., `normalized_homo`).
- `--vector_type`: Text vector type corresponding to the physical property.
- `--checkpoint_dir`: Directory containing the model checkpoint to load for testing.
- `--seed`: Random seed for reproducibility.
- `--output_file`: File to save testing results (default: `benchmark_results.txt`).

#### Example Command for Testing

```bash
python test.py --data_folder /path/to/test_data --csv_file /path/to/csv_file.csv \
--batch_size 1000 --physical normalized_homo --vector_type text_vector_homo \
--checkpoint_dir /path/to/checkpoints --seed 42 --output_file results.txt
```

### Output Files

The scripts will generate the following output files:

#### Training Output Files
Located in the specified `output_dir`:
- `train_loss.txt`: Training loss at each epoch.
- `test_loss.txt`: Validation loss at each epoch.
- `model_pred_list.txt`: Model predictions during validation.
- `model.pth`: Best-performing model weights.
- `train_duration.txt`: Total training time.

#### Testing Output File
- `benchmark_results.txt` (or the specified output file): Contains model parameters, mean test loss, and other relevant metrics.

## Model Overview

The **Pure2DopeNet** model comprises:
- **Convolutional Layers**: For processing image input.
- **Text Embedding Layers**: For integrating text vectors related to the target physical property.
- **Fully Connected Layers**: To combine image features and text embeddings for prediction.

For more details, see `model.py`.

## Reproducibility

To ensure consistent results, set the `--seed` argument in both training and testing scripts. This controls random processes such as data shuffling and model initialization.

---

For issues, questions, or contributions, feel free to reach out!
```
