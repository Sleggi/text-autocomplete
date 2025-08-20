# Проект автодополнения текста

В этом проекте реализована система автодополнения текста с использованием:

1. **Собственная LSTM-модель**, обученная на датасете коротких текстов (твитов).
2. **Предобученная трансформер-модель** (DistilGPT-2) для сравнения.

## Структура проекта

```
data/                # train.txt, val.txt, test.txt, raw_dataset.txt
models/              # Сохраненные веса LSTM
tokenizer/           # bpe_tokenizer.json
src/                 # Скрипты Python
solution.ipynb       # Jupyter notebook с обучением и оценкой
```

## Результаты

### Сравнение LSTM и предобученного трансформера (DistilGPT2)

#### 1. LSTM: метрики по эпохам

| Эпоха | Train Loss | Val Loss | ROUGE-1 |
|-------|------------|----------|---------|
| 1/10  | 6.7318     | 6.0905   | 0.1250  |
| 2/10  | 6.0621     | 5.9105   | 0.1315  |
| 3/10  | 6.0109     | 5.8629   | 0.1366  |
| 4/10  | 5.9073     | 5.8732   | 0.1410  |
| 5/10  | 5.9176     | 5.9312   | 0.1466  |
| 6/10  | 6.0484     | 5.9877   | 0.1503  |
| 7/10  | 6.0137     | 6.0257   | 0.1497  |
| 8/10  | 6.0844     | 6.0691   | 0.1551  |
| 9/10  | 6.2463     | 6.1460   | 0.1492  |
| 10/10 | 6.2086     | 6.1846   | 0.1544  |

**Примеры генерации LSTM после 10-й эпохи:**

- Prompt: *The weather today is amazing*  
  Generated: *the weather today is amazing that ' s , sad that has the eee had*

- Prompt: *I just watched a movie, it was*  
  Generated: *i just watched a movie , it was great do you have at ' ll to self and*

- Prompt: *Learning machine learning is fun and exciting*  
  Generated: *learning machine learning is fun and exciting will morning , won ' t some on my little*

---

#### 2. Предобученный трансформер DistilGPT2

**Примеры генерации на тех же промптах:**

- Prompt: *The weather today is amazing*  
  Generated: *The weather today is amazing. Weather data from the National Weather Service is available*

- Prompt: *I just watched a movie, it was*  
  Generated: *I just watched a movie, it was a lot of fun," he said. "I*

- Prompt: *Learning machine learning is fun and exciting*  
  Generated: *Learning machine learning is fun and exciting. It›s about teaching a new way*

**Примеры генерации на raw датасете твитов:**

- Prefix: *is upset that he can’t update his Facebook by texting it… and might cry as a resu*  
  Target: *lt  School today also. Blah!*  
  Generated: *ptor, but he’s still not sure how to respond to this.*  

- Prefix: *@Kenichan I dived many times for the ball. Managed to save 50%  Th*  
  Target: *e rest go out of bounds*  
  Generated: *umbs down from the field during the first half. Can’t get too close to the ball.*  

**Среднее значение ROUGE-1 на валидационном датасете:** `0.0476`

---

#### 3. Вывод

- LSTM показывает **лучший ROUGE-1 (≈0.15)** на валидационном наборе, чем предобученный DistilGPT2 (≈0.048) на тех же данных.  
- LSTM генерирует менее «связные» предложения по смыслу, но ближе к специфике тренировочного датасета.  
- Предобученный трансформер генерирует грамматически корректные и более «естественные» тексты, но не адаптирован к конкретной задаче автодополнения твитов.

> Примечания:
> - LSTM обучался на 10 эпохах на конкретном датасете твитов, поэтому его генерации более приближены к стилю тренировочных данных.  
> - Предобученный DistilGPT2 не дообучался на этом датасете, поэтому его генерации естественные, но менее похожи на конкретный набор данных.  
> - ROUGE-1 измеряет совпадение токенов с целевыми текстами; LSTM показывает выше метрику, потому что «учился» именно на этих твитах.  

---

