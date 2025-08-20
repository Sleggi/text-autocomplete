def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Выполняет один полный проход по датасету (одну эпоху) для обучения LSTM модели.

    1. Переводит модель в режим обучения.
    2. Для каждого батча:
       - обнуляет градиенты оптимизатора,
       - переносит данные на нужное устройство,
       - делает прямой проход через модель,
       - "разворачивает" выход и целевые значения для удобного вычисления потерь,
       - применяет маску, чтобы не учитывать паддинги при вычислении loss,
       - вычисляет loss, делает backpropagation и обновляет параметры модели.
    3. Накапливает loss для всех батчей и возвращает среднее значение.
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        x, y, mask = batch
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        # forward проход
        output, _ = model(x) 

        output = output.view(-1, output.size(-1))
        y_flat = y.view(-1)
        mask_flat = mask.view(-1)

        # применяем маску при учете loss
        loss = criterion(output[mask_flat], y_flat[mask_flat])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)