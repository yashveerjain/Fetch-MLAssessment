from torch.utils.data import Dataset, DataLoader
import torch

class TaskDataset(Dataset):
    def __init__(self, data=None, max_len=512):
        super(TaskDataset, self).__init__()
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
        # Assuming the dataset it tuple of (anchor, positive, negative)
        self.data = data
        self.vocab_size = self.tokenizer.vocab_size
        self.max_len = max_len

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def encode(self, text):
        return self.tokenizer(text, padding = True, truncation = True, max_length = self.max_len, return_tensors = 'pt')['input_ids']

    def __getitem__(self, idx):

        data = {
            'anchor': self.tokenizer(self.data[idx][0], padding = True, truncation = True, max_length = self.max_len, return_tensors = 'pt')['input_ids'],
            'positive': self.tokenizer(self.data[idx][1], padding = True, truncation = True, max_length = self.max_len, return_tensors = 'pt')['input_ids'],
            'negative': self.tokenizer(self.data[idx][2], padding = True, truncation = True, max_length = self.max_len, return_tensors = 'pt')['input_ids']}
        #  self.tokenizer(self.data[idx], padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
        return data