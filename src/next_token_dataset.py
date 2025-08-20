import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class NextTokenDataset(Dataset):
    """
    Датасет для задачи предсказания следующего токена.

    Каждый элемент датасета возвращает пару (x, y):
    - x: последовательность токенов без последнего токена
    - y: последовательность токенов, смещённая на 1 (следующие токены)
    """
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.tweets = [
            [int(tok) for tok in line.strip().split()]
            for line in lines
        ]

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tokens = self.tweets[idx]
        x = tokens[:-1]  # все токены кроме последнего
        y = tokens[1:]   # смещённые на 1
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_token=0):
    """
    Функция настройки маски и паддинга для DataLoader.

    1. Разворачивает батч на отдельные последовательности xs и ys.
    2. Применяет pad_sequence для выравнивания длины.
    3. Создаёт маску, которая игнорирует паддинг.
    """
    xs, ys = zip(*batch)
    
    # pad_sequence автоматически паддит до длины самой длинной последовательности в батче
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_token)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=pad_token)
    
    # маска: 1 для ненулевых токенов, 0 для паддинга
    mask = (xs_padded != pad_token).long()
    
    return xs_padded, ys_padded, mask


def create_dataloader(file_path, batch_size=256, shuffle=True):
    dataset = NextTokenDataset(file_path)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )


def get_all_dataloaders(train_path, val_path, test_path, batch_size=256):
    train_loader = create_dataloader(train_path, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(val_path, batch_size=batch_size, shuffle=False)
    test_loader  = create_dataloader(test_path, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader