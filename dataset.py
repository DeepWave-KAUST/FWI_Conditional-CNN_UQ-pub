import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(input_tensor, target_tensor, batch_size):
    """Makes a two-moon train-test dataset

    Parameters
    ----------
    train_size : :obj:`int`
        Number of training samples
    test_size : :obj:`int`
        Number of test samples
    batch_size : :obj:`int`, optional
        Batch size. If None, full batch
        
    Returns
    -------
    train_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    test_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Testing dataloader

    """
    # Create dataset
    data    = input_tensor
    label   = target_tensor
    dataset = TensorDataset(data, label)

    # Split into training (70%) and testing (30%) datasets
    train_size = int(0.7 * data.shape[0])
    test_size = int(data.shape[0]) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Use Pytorch's functionality to load data in batches.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False)    
        
    return train_loader, test_loader