import ast
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultimodalDataloader(Dataset):
    def __init__(self, data_folder, csv_file, predict_type, vector_type, transform=None):
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.transform = transform
        self.images = []
        self.molecule_data = {}
        self.predict_type = predict_type  # Assign predict_type as an instance variable
        self.vector_type = vector_type
        self._load_data()

    def _load_data(self):
        data_df = pd.read_csv(self.csv_file)
        for index, row in data_df.iterrows():
            dope_amount = row['dope_name']
            text_vector_type = torch.Tensor(ast.literal_eval(row[self.vector_type]))
            self.molecule_data[dope_amount] = {
                'predict_type': row[self.predict_type],
                'text_vector_type': text_vector_type
            }

        folders = os.listdir(self.data_folder)
        for folder in folders:
            if os.path.isdir(os.path.join(self.data_folder, folder)):
                images = os.listdir(os.path.join(self.data_folder, folder))
                for image in images:
                    self.images.append(os.path.join(self.data_folder, folder, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images[idx]
        molecule_name = os.path.basename(os.path.dirname(image_path))
        data = self.molecule_data[molecule_name]
        prediction = data['predict_type']
        text_vector = data['text_vector_type']
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, prediction, molecule_name, text_vector


