import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def transformer_evaluate(val_texts, num_samples=200):
    """
    Оценка качества предсказаний предобученной трансформер-модели на наборе текстов.

    Для каждого текста из val_texts берется префикс (75% текста), после чего модель
    генерирует продолжение (25%). Генерация выполняется с выборкой (sampling) с
    параметрами top-k и top-p. Далее вычисляется ROUGE-1 между сгенерированным
    продолжением и реальным текстом. Печатаются первые 3 примера для наглядности, а
    в конце выводится средний ROUGE-1.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    predictions, references = [], []
    
    for i, text in enumerate(val_texts[:num_samples]):
        if len(text) < 4:
            continue
        
        split_idx = int(len(text) * 0.75)
        prefix, target = text[:split_idx], text[split_idx:]
        
        inputs = tokenizer(prefix, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=len(target),
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        predictions.append(continuation)
        references.append(target)
        
        if i < 3:
            print("="*40)
            print("Prefix:", prefix)
            print("Target:", target)
            print("Generated:", continuation)
    
    rouge1_scores = [scorer.score(ref, pred)['rouge1'].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    
    print(f"\nAverage ROUGE-1: {avg_rouge1:.4f}")
    return avg_rouge1

def transformer_generate(texts):
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 10, 
                do_sample=True,     
                top_k=50,            
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"Prompt transformer: {text}")
            print(f"Generated transformer: {decoded_text}\n")