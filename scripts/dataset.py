import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SimpleImageCaptionDataset(Dataset):
    """CSV format: image_path,caption

    image_path may be absolute or relative to the CSV location.
    """

    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.rows = []
        with open(self.csv_path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            for r in reader:
                if not r:
                    continue
                # allow optional header: skip if header-like
                if r[0].lower().startswith('image'):
                    continue
                image_path = r[0]
                caption = r[1] if len(r) > 1 else ''
                self.rows.append((image_path, caption))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_path, caption = self.rows[idx]
        p = Path(image_path)
        if not p.is_absolute():
            p = (self.csv_path.parent / p)
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, caption
