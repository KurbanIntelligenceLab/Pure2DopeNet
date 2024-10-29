import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import MultimodalDataloader
from model import Pure2DopeNet


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        images, molecule_energies, molecule_names, text_vectors = data
        inputs = images.to(device)
        labels = molecule_energies.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, text_vectors.to(device))
        outputs = outputs.view(-1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    output = torch.zeros(0, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, molecule_energies, molecule_names, text_vectors = data
            inputs = images.to(device)
            labels = molecule_energies.to(device)
            outputs = model(inputs, text_vectors.to(device))
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            output = torch.cat((output, outputs), 0)

        output = torch.mean(output)
        labels = labels[0]
    return running_loss / len(test_loader), output.to('cpu'), labels


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Pure2DopeNet()

    model_name = model.__class__.__name__
    predict_type = args.physical
    vector_type = args.vector_type

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    custom_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root = args.root
    data_folder = args.data_folder
    csv_file = args.csv_file
    model_result_path = f"{args.output_dir}/{predict_type}/{model_name}/{args.seed}"

    train_dataset = MultimodalDataloader(f"{data_folder}/train", csv_file, predict_type, vector_type,
                                         custom_transforms)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                                                                 [int(len(train_dataset) * 0.8),
                                                                  int(len(train_dataset) * 0.2)])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              generator=torch.manual_seed(args.seed))
    test_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                             generator=torch.manual_seed(args.seed))

    model.to(device)
    reset_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loss = []
    val_loss = []
    model_pred_list = []
    fold_duration = []
    fold_start_time = time.time()

    for epoch in tqdm(range(args.num_epochs)):
        print(f"Epoch {epoch + 1} of {args.num_epochs}")
        train_epoch_loss = train(model, train_loader, optimizer, criterion, device)
        test_epoch_loss, model_pred, label = validate(model, test_loader, criterion, device)
        lr_scheduler.step()
        train_loss.append(train_epoch_loss)
        val_loss.append(test_epoch_loss)
        model_pred_list.append(model_pred)
        print(f"Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}")

        if not os.path.exists(f'{root}/{model_result_path}'):
            os.makedirs(f'{root}/{model_result_path}')

        np.savetxt(f'{root}/{model_result_path}/train_loss.txt', train_loss)
        np.savetxt(f'{root}/{model_result_path}/test_loss.txt', val_loss)
        np.savetxt(f'{root}/{model_result_path}/model_pred_list.txt', model_pred_list)

        if test_epoch_loss <= min(val_loss):
            torch.save(model.state_dict(), f"{root}/{model_result_path}/model.pth")
            print("Saved best model weights!")

    fold_end_time = time.time()
    fold_duration.append(fold_end_time - fold_start_time)
    np.savetxt(f'{root}/{model_result_path}/train_duration.txt', fold_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pure2DopeNet model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--root", type=str, default="folder_root", help="Root folder for saving results")
    parser.add_argument("--data_folder", type=str, default="image_root", help="Data folder containing images")
    parser.add_argument("--csv_file", type=str, default="csv_file.csv", help="CSV file with data labels")
    parser.add_argument("--output_dir", type=str, default="training", help="Directory for output results")
    parser.add_argument("--seed", type=int, default=5, help="Random seed for reproducibility")
    parser.add_argument("--physical", type=str, required=True,
                        help="Physical property to predict (e.g., normalized_homo)")
    parser.add_argument("--vector_type", type=str, required=True,
                        help="Type of text vector corresponding to the physical property")

    args = parser.parse_args()
    main(args)
