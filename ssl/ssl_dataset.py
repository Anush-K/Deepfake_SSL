import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
from ssl.augmentations import get_ssl_transforms


class SSLDataset(Dataset):
    def __init__(self, csv_files, image_size=224):

        self.df = pd.concat([pd.read_csv(f) for f in csv_files])
        self.df = self.df[self.df["split"] != "test"].reset_index(drop=True)
        self.base_aug, self.highpass_aug = get_ssl_transforms(image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        path = self.df.iloc[idx]["image_path"]
        img = Image.open(path).convert("RGB")

        view1 = self.base_aug(img)

        if random.random() < 0.5:
            view2 = self.highpass_aug(img)
        else:
            view2 = self.base_aug(img)

        return view1, view2