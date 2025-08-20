import re
import emoji
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Lowercase
from pathlib import Path
from sklearn.model_selection import train_test_split


def clean_dataset(input_file: str, output_file: str):
    """
    Очищает датасет от лишних символов, ссылок, упоминаний, эмодзи, 
    и сохраняет результат в новый файл.

    1. Приведение текста к нижнему регистру.
    2. Удаление URL (http, https, www).
    3. Удаление упоминаний пользователей (@username).
    4. Удаление всех эмодзи.
    5. Удаление текстовых эмодзи ;d :P и т.д.
    6. Обработка повторяющихся знаков пунктуации ( "...." -> ".").
    7. Удаление нестандартных символов.
    8. Удаление лишних пробелов.
    """
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = emoji.replace_emoji(text, replace="")  
       
        emoticon_pattern = r'[:;=8][\-o\*]?[)\(dDpP]'
        text = re.sub(emoticon_pattern, "", text)
        text = re.sub(r'([.!?,;:])\s+\1+', r'\1', text)
        text = re.sub(r'([.!?,;:])\1+', r'\1', text)
        text = re.sub(r"[^a-z0-9\s.,!?;:()'\"]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            cleaned = clean_text(line)
            if cleaned:
                f_out.write(cleaned + "\n")
    print(f"Очищенный датасет сохранен - {output_file}")

def tokenize_dataset(input_file: str, output_file: str, vocab_size: int = 20000):
    """
    Токенизирует текстовый датасет с помощью BPE (Byte Pair Encoding) и сохраняет
    как сам токенизатор, так и токенизированный файл.

    1. Обучение токенизатора BPE на исходном файле.
       - Используется нормализация в нижний регистр.
       - Предтокенизация по пробелам.
       - Минимальная частота токена = 2.
       - В словарь добавляются специальные токены: [PAD], [UNK], [CLS], [SEP], [MASK].
    2. Токенизация исходного файла:
       - Каждая строка (текст/сообщение) преобразуется в последовательность индексов токенов.
       - Результат сохраняется в `output_file`, токены разделяются пробелами.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train([input_file], trainer)

    # Сохраняем токенайзер
    output_dir = Path("tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "bpe_tokenizer.json"))

    # Токенизируем файл
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            encoding = tokenizer.encode(line)
            token_ids = " ".join(map(str, encoding.ids))
            f_out.write(token_ids + "\n")
    print(f"Токенизированный датасет сохранен в - {output_file}")

def split_dataset(input_file: str, train_file: str, val_file: str, test_file: str):
    """
    Разбивает токенизированный датасет на тренировочную, валидационную и тестовую выборки
    и сохраняет их в отдельные файлы.

    1. Загружает все строки из исходного файла `input_file`.
    2. Делит данные на:
       - 80% для тренировки (train)
       - 10% для валидации (val)
       - 10% для теста (test)
       Используется `train_test_split` с фиксированным `random_state=42` для воспроизводимости.
    3. Сохраняет каждую выборку в отдельный файл: `train_file`, `val_file`, `test_file`.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    train, temp = train_test_split(lines, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for path, data in zip([train_file, val_file, test_file], [train, val, test]):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(data)

    print(f"Соотношение датасетов - {len(train)} train, {len(val)} val, {len(test)} test")


if __name__ == "__main__":
    clean_dataset("data/raw_dataset.txt", "data/processed_dataset.txt")
    tokenize_dataset("data/processed_dataset.txt", "data/tokenized_dataset.txt")
    split_dataset(
        "data/tokenized_dataset.txt",
        "data/train.txt",
        "data/val.txt",
        "data/test.txt"
    )