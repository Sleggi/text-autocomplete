import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    LSTM модель для задачи предсказания следующего токена.
    
    Архитектура:
        - Embedding слой для токенов
        - LSTM слой
        - Dropout
        - Линейный слой для проекции в размер словаря
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len] — батч токенов
        hidden: начальные hidden/cell состояния LSTM
        """
        # Преобразуем токены в embedding
        x = self.embedding(x)
        
        # Прогон через LSTM
        output, hidden = self.lstm(x, hidden)
        
        output = self.dropout(output)
        
        # Получает логиты
        logits = self.fc(output)
        
        return logits, hidden
    
    @torch.no_grad()
    def generate(self, prompt, max_len=1, temperature=1.0):
        """
        Генерация последовательности токенов
        prompt: [batch_size, seq_len] или [seq_len] - стартовая последовательность токенов
        max_len: сколько токенов сгенерировать
        temperature: сглаживание вероятностей (чем выше, тем более случайно)
        """
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        generated = prompt.clone()
        hidden = None

        for _ in range(max_len):
            logits, hidden = self.forward(generated[:, -1:], hidden)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated