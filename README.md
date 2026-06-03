# 🏷️ Peso das Palavras

> Pipeline de NLP para predição de nota de satisfação (1–5) a partir de comentários de clientes em marketplace — do pré-processamento clássico com TF-IDF até embeddings contextuais com BERTimbau e classificação de sentimento com XLM-RoBERTa.

**Disciplina:** PII 3 — Prof. Dr. Felipe A. Camargo  
**Parceria:** AWS  
**Dataset:** ~99 mil pedidos · ~41 mil avaliações com comentário de texto

---

## Problema de negócio

Em marketplaces digitais, clientes nem sempre atribuem notas coerentes com o que escreveram — ou simplesmente não avaliam numericamente. O objetivo deste projeto é construir um modelo capaz de **estimar a nota de satisfação (1 a 5) com base apenas no texto do comentário**, sem depender da nota declarada pelo usuário.

---

## Estrutura do projeto

```
peso-das-palavras/
│
├── notebooks/
│   ├── etapa1_tratamento_eda.ipynb        # Tipagem, limpeza, EDA
│   ├── etapa2_tfidf_ridge.ipynb           # Baseline TF-IDF + Ridge
│   ├── etapa3_bertimbau_sentimento.ipynb  # BERTimbau + features de sentimento
│   └── etapa4_classificacao_sentimento.ipynb  # XLM-RoBERTa
│
├── data/
│   ├── raw/              # Arquivos originais (não versionados)
│   └── processed/
│       └── dataset_tratado.csv
│
├── embeddings/
│   └── embeddings_bertimbau.npy   # Gerado na Etapa 3 (~34min com GPU)
│
└── README.md
```

---

## Pipeline

### Etapa 1 — Pré-processamento e EDA

Tratamento das tabelas `pedidos.csv` e `avaliacoes.csv` com join por `order_id`.

**Tipagem:**
- Colunas de data: `object` → `datetime64`
- IDs: `object` → `StringDtype`
- `order_status`: `object` → `category`
- `review_score`: `float` → `Int64`

**Limpeza:**
- Filtro: apenas pedidos com `order_status == 'delivered'` e com `order_delivered_customer_date` preenchida
- Remoção de duplicatas por `order_id` e `review_id`

**Features derivadas:**

| Feature | Origem | Descrição |
|---|---|---|
| `dias_atraso` | pedidos | `data_entrega_real − data_estimada` (negativo = adiantado) |
| `atrasado` | pedidos | flag booleana — `dias_atraso > 0` |
| `tem_comentario` | avaliacoes | flag booleana — comentário não nulo |
| `horas_resposta` | avaliacoes | horas entre criação e resposta da avaliação |

**Output:** `dataset_tratado.csv`

---

### Etapa 2 — Baseline: TF-IDF + Ridge Regression

Transforma cada comentário em um vetor numérico via TF-IDF e treina uma Ridge Regression.

**Por que Ridge e não OLS?**  
Com ~8.000 features (palavras), a regressão linear padrão sofre overfitting. A Ridge adiciona uma penalidade L2 aos coeficientes, forçando o modelo a generalizar:

```
Ridge: minimiza  Σ(y − ŷ)²  +  α × Σ(coef²)
                               ↑ penalidade (α = 1.0)
```

**Parâmetros TF-IDF:**

| Parâmetro | Valor |
|---|---|
| `max_features` | 8.000 |
| `ngram_range` | (1, 2) |
| `min_df` | 20 |
| `sublinear_tf` | True |
| `strip_accents` | unicode |

---

### Etapa 3 — BERTimbau + Features de Sentimento + Ridge

**Hipótese:** embeddings contextuais capturam negação, ironia e semântica que o TF-IDF não representa.

**Modelo base:** [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (BERTimbau) — BERT treinado em português brasileiro.

Cada comentário é convertido em um vetor de 768 dimensões (embedding do token `[CLS]`). Os embeddings são gerados em GPU (~34 min para ~41k comentários) e salvos em `embeddings_bertimbau.npy`.

**Combinações testadas:**

| Modelo | Features |
|---|---|
| TF-IDF | ~8.000 palavras |
| BERTimbau | 768 embeddings |
| BERT + TF-IDF | 768 + ~8.000 |
| BERT + TF-IDF + VADER | + score VADER |
| BERT + TF-IDF + LeIA | + score LeIA (PT-BR) |
| BERT + TF-IDF + TextBlob | + polaridade TextBlob |
| BERT + TF-IDF + Todos | todas as features acima |

**Por que o sentimento teve pouco impacto?**  
Os embeddings do BERTimbau já capturam grande parte das informações de polaridade. A diferença entre `gostei` e `não gostei` está representada nos vetores contextuais — não sendo necessário explicitar via VADER, LeIA ou TextBlob.

---

### Etapa 4 — Classificação de Sentimento com XLM-RoBERTa

Classifica cada comentário em **positivo**, **negativo** ou **neutro** usando um modelo multilíngue pré-treinado, sem fine-tuning.

**Modelo:** [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) — treinado em ~198 milhões de tweets em múltiplos idiomas.

**Mapeamento nota → sentimento (ground truth):**

| Nota | Sentimento |
|---|---|
| 1–2 | negativo |
| 3 | neutro |
| 4–5 | positivo |

Os casos onde o modelo de sentimento **diverge** da nota são os mais ricos: revelam clientes que escreveram algo positivo mas deram nota baixa (ou vice-versa), sinalizando oportunidades de intervenção para equipes de atendimento.

---

## Resultados

### Predição de nota (Etapa 3)

| Modelo | Pearson r | Spearman ρ | RMSE | MAE | Acurácia ±1 |
|---|---|---|---|---|---|
| TF-IDF + Ridge | — | — | — | — | — |
| BERTimbau + Ridge | — | — | — | — | — |
| **BERT + TF-IDF + Ridge** | **0,8427** | **0,7435** | **0,8451** | **0,5891** | **91,71%** |
| BERT + TF-IDF + LeIA | 0,8427 | 0,7435 | 0,8449 | 0,5890 | 91,71% |

> A diferença entre BERT + TF-IDF e BERT + TF-IDF + LeIA foi de **0,0002 RMSE** — sem impacto prático.  
> Modelo final escolhido: **BERT + TF-IDF + Ridge** (melhor equilíbrio entre desempenho, simplicidade e interpretabilidade).

### Validação estatística

Teste de Wilcoxon comparando TF-IDF puro vs BERT + TF-IDF:

```
Estatística : 9.496.187,50
p-value     : 4,35 × 10⁻⁷¹
```

A melhoria do modelo híbrido é **estatisticamente significativa** (p << 0,05).

---

## Como executar

### Pré-requisitos

```bash
pip install pandas numpy scikit-learn torch transformers scipy matplotlib seaborn
pip install vaderSentiment textblob leia
```

### Ordem de execução

```
1. etapa1_tratamento_eda.ipynb       → gera dataset_tratado.csv
2. etapa2_tfidf_ridge.ipynb          → baseline TF-IDF
3. etapa3_bertimbau_sentimento.ipynb → gera embeddings_bertimbau.npy + modelo final
4. etapa4_classificacao_sentimento.ipynb → classificação positivo/negativo/neutro
```

> **Atenção:** A geração dos embeddings (Etapa 3) requer GPU e leva ~34 minutos para ~41k comentários. Se os embeddings já estiverem salvos em `embeddings_bertimbau.npy`, o notebook os carrega automaticamente.

### Predição ao vivo

```python
nota = predict_final("Produto idêntico à foto, chegou antes do prazo!")
# → 5/5 ⭐⭐⭐⭐⭐
```

---

## Stack

| Categoria | Ferramentas |
|---|---|
| Manipulação de dados | `pandas`, `numpy` |
| NLP clássico | `scikit-learn` (TF-IDF, Ridge) |
| NLP contextual | `transformers`, `torch` (BERTimbau, XLM-RoBERTa) |
| Sentimento | `vaderSentiment`, `leia`, `textblob` |
| Visualização | `matplotlib`, `seaborn` |
| Estatística | `scipy` (Wilcoxon) |
| Ambiente | Google Colab (GPU T4) |

---

## Referências

- Souza, F., Nogueira, R., & Lotufo, R. (2020). *BERTimbau: Pretrained BERT models for Brazilian Portuguese*. [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased)  
- Barbieri, F. et al. (2022). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis*. [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)  
- Hutto, C. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*.
