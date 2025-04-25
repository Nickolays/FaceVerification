from PIL import Image
import os
from torch.utils.data import Dataset
import pandas as pd


class RecognitionDataset(Dataset):
    def __init__(self, main_path, pair_path, is_train=False, transformations=None):
        super().__init__()
        assert os.path.exists(main_path)
        assert os.path.exists(pair_path)

        self.main_path = main_path
        self.transformations = transformations
        self.is_train = is_train
        
        # Load pairs from CSV
        self.all_pairs = pd.read_csv(pair_path)  # We have 3 columns here: 'face1', 'face2', 'target'
        # Apply sampling/filtering logic
        # self._get_dataset()
        # Shuffle pairs initially
        self.on_epoch_end()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """    """
        row = self.pairs.iloc[idx]
        face1 = Image.open(os.path.join(self.main_path, row['face1']))
        face2 = Image.open(os.path.join(self.main_path, row['face2']))
        target = row['target']
        if self.transformations:
            face1 = self.transformations(face1)
            face2 = self.transformations(face2)
            
        return {
            'face1': face1,
            'face2': face2,
            'target': target
        }
    
    def _get_dataset(self):
        # Balance positive and negative samples
        positiv = self.pairs[self.pairs['target'] == 1]
        negativ = self.pairs[self.pairs['target'] == 0]

        # Optional: sample negatives to balance
        # negativ = negativ.sample(len(positiv))  # Uncomment if needed
        self.pairs = pd.concat([positiv, negativ], ignore_index=True)
        # print("Target distribution:\n", self.pairs['target'].value_counts())

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        # Balance positive and negative samples
        if self.is_train:
            positiv = self.all_pairs[self.all_pairs['target'] == 1].sample(48000)
            negativ = self.all_pairs[self.all_pairs['target'] == 0].sample(48000)
        else:
            positiv = self.all_pairs[self.all_pairs['target'] == 1].sample(18000)
            negativ = self.all_pairs[self.all_pairs['target'] == 0].sample(18000)

        self.pairs = pd.concat([positiv, negativ], ignore_index=True)
        # Shuffle
        self.pairs = self.pairs.sample(frac=1).reset_index(drop=True)

    
def prepare_csv(path_to_images:str, save_path:str):
    """ """
    import itertools
    import numpy as np

    faces = os.listdir(path_to_images)
    # Сочетания – выбранные из множества n объектов комбинации m объектов, отличающиеся хотя бы одним объектом. Порядок элементов не важен.
    all_combination = pd.DataFrame(
        list(itertools.combinations(faces, 2)),
        columns=('face1', 'face2')
    )
    all_combination['name1'] = all_combination['face1'].apply(lambda x: x.split('_')[0])
    all_combination['name2'] = all_combination['face2'].apply(lambda x: x.split('_')[0])
    # 
    all_combination['target'] = np.where(all_combination['name1'] == all_combination['name2'], 1, 0)
    # Save to .csv file
    all_combination.to_csv(save_path, index=False)
    print(f"CSV saved to {save_path}")