---
library_name: transformers
language:
- pt
metrics:
- accuracy
base_model:
- distilbert/distilbert-base-uncased
pipeline_tag: text-classification
---

# SYAS1

SYAS1-PTBR é um modelo Transformer com base no Distilbert, focado em análise de sentimentos para o Português Brasileiro. 

O principal intuito dele, além de estudo, é contribuir com a comunidade brasileira, pois há uma grande escassez de LLMs para a lingua portuguesa.

## Descrição

O SYAS1-PTBR é um modelo baseado no Distilbert-base sendo treinado por fine-tuning utilizando o dataset "Portuguese Tweets for Sentiment Analysis" do Kaggle. O seu treinamento foi feito exclusivamente pelo Google Colab Pro, utilizando de uma GPU NVIDIA L4.

## Detalhes do Modelo

- **Tipo do Modelo:** Text Classification
- **Licença:** Apache 2.0
- **Modelo base:** distilbert-base-uncased
- **Idioma:** Português Brasileiro

## Como usar

```Python

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("1Arhat/SYAS1-PTBR")
model = AutoModelForSequenceClassification.from_pretrained("1Arhat/SYAS1-PTBR")

# Texto para classificação
texto = "Esse produto é incrível! Recomendo muito."

# Processar seu texto
inputs = tokenizer(texto, return_tensors="pt")

# Previsões do modelo

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicao = torch.argmax(logits, dim=1).item()

#print(f"Classe prevista: {predicao}")

# Output: Classe prevista: 2

# Caso deseje que ele entregue porcentagens de todas as classes:

probs = F.softmax(logits, dim=1)  # Uso da função softmax para transformar a saída do logits em probabilidades

labels = ["Negativo", "Neutro", "Positivo"]

for idx, label in enumerate(labels):
    print(f'{label}: {probs[0][idx]}')

# Output:

#Negativo: 0.2401905506849289
#Neutro: 0.028042761608958244
#Positivo: 0.7317667007446289

# Caso deseje mais praticidade, use o pipeline do Hugging face
```

## Função

O SYAS1 foi criado para análise de sentimentos com foco na língua portuguesa.

Por ser um modelo treinado por fine-tuning do DistilBERT, ele pode funcionar em inglês, mas é altamente recomendado utilizá-lo apenas para o português.

Se precisar de análise de sentimentos em outro idioma, recomendo usar outros modelos disponíveis.

### Resultados
| Metric          | Value | 
|-----------------|-------|
| Accuracy        | 0.7384| 
| F1-SCORE        | 0.74  |