import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import MultimodalDataloader
from model import Pure2DopeNet


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    output = torch.zeros(0, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, molecule_energies, molecule_names, text_vector = data
            inputs = images.to(device)
            labels = molecule_energies.to(device)

            outputs = model(inputs, text_vector.to(device))
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            output = torch.cat((output, outputs), 0)

        output = torch.mean(output)
        labels = labels[0]
    return running_loss / len(test_loader), output.to('cpu'), labels


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_vs_avg_test_loss = {}
    model = Pure2DopeNet()

    custom_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    criterion = nn.L1Loss()

    model_name = model.__class__.__name__
    predict_type = args.physical
    vector_type = args.vector_type

    test_dataset = MultimodalDataloader(f"{args.data_folder}/test", args.csv_file, predict_type, vector_type,
                                        custom_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    chkpt_path = f'{args.checkpoint_dir}/{predict_type}/{model_name}/{args.seed}/model.pth'
    model.load_state_dict(torch.load(chkpt_path, map_location=device))
    model.to(device)

    test_loss, output, labels = test(model, test_loader, criterion, device)
    print(f"Test loss for seed {args.seed}: {test_loss}")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_vs_avg_test_loss[model_name] = [model_params, test_loss]

    with open(args.output_file, 'a') as f:
        f.write(f'Physical: {predict_type} Model: {model_name} params, mse: {model_params}, {test_loss}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pure2DopeNet model")
    parser.add_argument("--data_folder", type=str, required=True, help="Data folder containing test images")
    parser.add_argument("--csv_file", type=str, required=True, help="CSV file with data labels")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for testing")
    parser.add_argument("--physical", type=str, required=True,
                        help="Physical property to predict (e.g., normalized_homo)")
    parser.add_argument("--vector_type", type=str, required=True,
                        help="Type of text vector corresponding to the physical property")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to load model checkpoints")
    parser.add_argument("--seed", type=int, default=5, help="Random seed for reproducibility")
    parser.add_argument("--output_file", type=str, default="benchmark_results.txt",
                        help="File to save benchmark results")

    args = parser.parse_args()
    main(args)
