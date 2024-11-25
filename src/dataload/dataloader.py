from torch.utils.data import DataLoader

from tokenizers.bytepair_tokenizer import BytePairTokenizer
from dataload.dataset import SimpleDataset

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = BytePairTokenizer()
    dataset = SimpleDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader