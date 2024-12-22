import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

# generally dataset
"""
    -Args:
        inputs: (data_len, seq_len, input_dim), targets: (data_len, seq_len)
    
    -Usage:
        dataset = GdataLoader(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)   
"""
class GDataSet(Dataset):
    def __init__(self, data_path: str, seq_len: int, is_normalization: bool, tar_col: str) -> None:
        self.seq_len = seq_len
        self.inputs, self.targets = self.split_dataset(data_path, is_normalization, self.seq_len, tar_col)

    # normalize the data
    def normalize_column(self, column_data: pd.DataFrame, is_save_max_min_val: bool):
        min_val = min(column_data)
        max_val = max(column_data)
        if is_save_max_min_val:
            self.min_val, self.max_val = min_val, max_val
        return [(x - min_val) / (max_val - min_val) for x in column_data]
    
    # denormalize the data
    def tar_denormalize(self, normalized_data):
        normalized_data = torch.tensor(normalized_data) if not isinstance(normalized_data, torch.Tensor) else normalized_data
        if self.min_val is None:
            raise ValueError("Min values is None.")
        if self.max_val is None:
            raise ValueError("Max values is None.")
        return normalized_data * (self.max_val - self.min_val) + self.min_val

    def split_dataset(self, data_path: str, is_normalization: bool, seq_len: int, tar_col: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = pd.read_csv(data_path, encoding='utf-8')
        if is_normalization:
            for col in data.columns:
                if col == tar_col:
                    data[col] = self.normalize_column(data[col], True)
                    continue
                data[col] = self.normalize_column(data[col], False) # convert to the normalized data e.g. [19, 20, 21] -> [0.8, 0.9, 1]

        assert seq_len < len(data), f"error seq_len len: {seq_len}, data len: {len(data)}"
        assert tar_col in data.columns, f"Target column '{tar_col}' not found in the dataset."

        inputs, targets = [], []
        for i in range(seq_len, len(data)):
            inputs.append(data.iloc[i - seq_len : i, :].values)
            targets.append(data.iloc[i - seq_len : i][tar_col]) # tar_col is your target col_name
        inputs, targets = np.array(inputs), np.array(targets)
        inputs, targets = torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
        return inputs, targets

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]
    
# figure dataset
"""
    -Req:
        folder stucture should be:
        /dataset
            /class_0
                image1.jpg
                image2.jpg
            /class_1
                image1.jpg
                image2.jpg
            /class_2
                image1.jpg
                image2.jpg
    -Args:
        root_dir: folder name, transform: define any transformations you need
    -Usage:
        transform = Compose([transforms.ToTensor(), transforms.Resize((width, height))]) e.g. torch.Size([3, 1603, 1621]) -> torch.Size([3, 128, 128]) # ToTensor() -> [0, 1]
        dataset = FolderDataset(root_dir='./dataset', transform=transform)
        image, label = dataset[0]
    -Return:
        Tuple[torch.Tensor, str], image: torch.Tensor, label: image className idx
"""

class FigureDataSet(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        assert os.path.isdir(root_dir), f"Error: The directory: {root_dir} does not exist"
        
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir)) # get the class name e.g. ['class_0', 'class_1']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)} # {'class_0': 0, 'class_1': 1}

        self.image_paths = [] # save image path
        self.labels = [] # save class name idx

        for class_name in self.class_names:
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    if img_path.endswith(('.jpg', '.png')): # filter img .jpg or .png
                        self.image_paths.append(img_path) # add the image path
                        self.labels.append(self.class_to_idx[class_name]) # add image relative lable
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    # from utilis import apply_transform
    # @apply_transform # use for diffusion transformer, convert the [0, 1] to [-1, 1]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB') # get image info
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# txt words and sentense dataset
"""
    -Args:
        text_file_path: corpus file path
        block_size: split the corpus file to each block_size
        seq_len: seq_len

    :dataset
    -Usage:
        dataset = LanguageDataLoader('beginning_gpt2/input_.txt', seq_len=10)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    -Return:
        Tuple[torch.Tensor, torch.Tensor], input_seq: (i: i+seq_len), tar_seq: (i+1: i+seq_len+1)
    
    :dataset.decode
    -Usage:
        decoder = dataset.decode(torch.tensor)
    -Return:
        str, decode str from vocabulary
"""
import string
from collections import Counter
import torch
from torch.utils.data import Dataset
class LanguageDataSet(Dataset):
    def __init__(self, text_file_path: str, block_size: int = 1024, seq_len: int = 100) -> None:
        self.seq_len = seq_len
        self.corpus = self.build_word_list(text_file_path, block_size)
        self.data_pairs = self.create_data_pairs(self.corpus)
    
    # remove all punctuation and convert to lowercase
    def preprocess_punctuation(self, texts: str) -> str:
        texts = texts.lower()
        texts = texts.translate(str.maketrans("", "", string.punctuation)) # remove all punctuation
        return texts

    # tokenizer the text to words
    def tokenizer(self, texts: str) -> str:
        return texts.split()
    
    # build the word list 
    def build_word_list(self, text_file_path: str, block_size: int = 1024) ->list:
        assert os.path.exists(text_file_path), f"text_file_path not exist"
        buffer = "" # store the incomplete word from previous block
        word_list = []
        # read file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            while True:
                # read to the block
                texts = f.read(block_size)
                if not texts:
                    break
                texts = buffer + texts
                texts = self.preprocess_punctuation(texts) # convert to lowrcase and clean punctuation
                words = self.tokenizer(texts)

                if texts.endswith(" "):
                    buffer = ""
                else:
                    buffer = words[-1] if len(words) > 0 else ""
                    words = words[:-1]
                word_list.extend(words)
        return  word_list

    def create_data_pairs(self, corpus: list[str]):
        data_pairs = [] # store the data_pairs
        counter = Counter(corpus) # save each word appear 
        self.vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), 1)}
        self.vocab['<PAD>'] = 0  # Add padding token
        self.vocab['<UNK>'] = len(self.vocab)  # Add unknown token
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        tokens = torch.tensor([self.vocab.get(token, self.vocab['<PAD>']) for token in corpus], dtype=torch.long)
        
        # split the token to (seq_len)
        assert self.seq_len < len(corpus), f"seq_len: {self.seq_len} can match the corpus size: {len(corpus)}"
        for i in range(len(corpus) - self.seq_len):
            input_seq = tokens[i:i + self.seq_len]
            tar_seq = tokens[i + 1:i + self.seq_len + 1]
            data_pairs.append((input_seq, tar_seq))
        
        return data_pairs
    
    # decode a tensor of token IDs back to a string using the reverse vocabulary.
    def decode(self, token_ids: torch.Tensor) -> str:
        words = [self.reverse_vocab.get(idx.item(), '<UNK>') for idx in token_ids]
        return ' '.join(words)

    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, tar_seq = self.data_pairs[idx]
        return input_seq, tar_seq