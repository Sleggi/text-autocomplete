import torch
from rouge_score import rouge_scorer

def lstm_evaluate(model, val_loader, criterion, device, idx2word=None):
    """
    Выполняет оценку модели на валидационном наборе данных.

    1. Переводит модель в режим оценки (eval).
    2. Для каждого батча:
       - переносит данные на нужное устройство,
       - делает прямой проход через модель,
       - вычисляет loss с маской, чтобы не учитывать паддинги,
       - если idx2word передан, декодирует предсказания и цели в текст и вычисляет ROUGE-1.
    3. Возвращает средний loss и средний ROUGE по всем батчам.
    """
    model.eval()
    total_loss = 0
    rouge_f1_list = []

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    with torch.no_grad():
        for batch in val_loader:
            x, y, mask = batch 
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            output, _ = model(x)

            output_flat = output.view(-1, output.size(-1))
            y_flat = y.view(-1)
            mask_flat = mask.view(-1)

            loss = criterion(output_flat[mask_flat], y_flat[mask_flat])
            total_loss += loss.item()
            
            if idx2word is not None:
                pred_tokens = torch.argmax(output, dim=-1)
                for pred_seq, target_seq, mask_seq in zip(pred_tokens, y, mask):
                    pred_text = " ".join(
                        [idx2word[idx.item()] for idx, m in zip(pred_seq, mask_seq) if m]
                    )
                    target_text = " ".join(
                        [idx2word[idx.item()] for idx, m in zip(target_seq, mask_seq) if m]
                    )
                    score = scorer.score(target_text, pred_text)
                    rouge_f1_list.append(score['rouge1'].fmeasure)

    avg_loss = total_loss / len(val_loader)
    avg_rouge = sum(rouge_f1_list) / len(rouge_f1_list) if rouge_f1_list else 0.0
    return avg_loss, avg_rouge

def lstm_generate(texts, tokenizer, model, device, max_len=10):
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer.encode(text)
            input_ids = torch.tensor([encoding.ids], device=device)

            generated = model.generate(input_ids, max_len=max_len)  
            decoded_text = tokenizer.decode(
                generated[0].tolist(), 
                skip_special_tokens=True
            )

            print(f"Prompt lstm: {text}")
            print(f"Generated lstm: {decoded_text}\n")