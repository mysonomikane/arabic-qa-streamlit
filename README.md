# 📚 Rapport de Projet NLP
## Système de Question-Réponse en Arabe basé sur Wikipedia

---

## 📋 Informations Générales

| Élément | Détail |
|---------|--------|
| **Projet** | Système RAG de Question-Réponse en Arabe |
| **Auteur** | sonomikane |
| **Date** | Janvier 2026 |
| **Modèle HuggingFace** | [sonomikane/arabert-qa-arabic-wikipedia](https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia) |
| **Application Streamlit** | [GitHub Repository](https://github.com/mysonomikane/arabic-qa-streamlit) |

---

## 1. 🎯 Objectif du Projet

Développer un **assistant intelligent** capable de répondre aux questions en arabe en utilisant Wikipedia arabe comme base de connaissances.

### Architecture RAG (Retrieval-Augmented Generation)

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Question      │───▶│  🔍 RETRIEVAL        │───▶│  📚 Wikipedia   │
│   utilisateur   │    │  Recherche Wikipedia │    │  Arabe          │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  Contexte pertinent  │
                       │  (articles trouvés)  │
                       └──────────────────────┘
                                  │
                                  ▼
┌─────────────────┐    ┌──────────────────────┐
│   📝 Réponse    │◀───│  🤖 GENERATION       │
│   extraite      │    │  AraBERT QA Model    │
└─────────────────┘    └──────────────────────┘
```

---

## 2. 🤖 Modèle de Base

### AraBERT-v2

| Caractéristique | Valeur |
|-----------------|--------|
| **Nom** | `aubmindlab/bert-base-arabertv2` |
| **Type** | BERT pré-entraîné pour l'arabe |
| **Paramètres** | ~135 millions |
| **Vocabulaire** | 64,000 tokens |
| **Créateur** | AUB MIND Lab |

---

## 3. 📊 Datasets d'Entraînement

### Datasets Utilisés

| Dataset | Type | Taille Train | Taille Val |
|---------|------|--------------|------------|
| **TyDi QA Arabic** | Question-Réponse | ~14,000 | ~1,500 |
| **ARCD** | Arabic Reading Comprehension | ~2,500 | ~700 |
| **XQuAD Arabic** | Cross-lingual QA | - | ~1,100 |

### Total après preprocessing

| Split | Nombre d'exemples |
|-------|-------------------|
| **Train** | 17,337 |
| **Validation** | 2,963 |

---

## 4. ⚙️ Configuration d'Entraînement

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| **Epochs** | 3 |
| **Learning Rate** | 3e-5 |
| **Batch Size (Train)** | 16 |
| **Batch Size (Eval)** | 32 |
| **Max Sequence Length** | 384 |
| **Stride** | 128 |
| **Warmup Ratio** | 0.1 |
| **Weight Decay** | 0.01 |
| **FP16** | Oui (Mixed Precision) |

### Environnement

| Élément | Spécification |
|---------|---------------|
| **Plateforme** | Google Colab |
| **GPU** | Tesla T4 (16 GB) |
| **Temps d'entraînement** | ~25.5 minutes |

---

## 5. 📈 Résultats

### Métriques d'Évaluation

| Métrique | Score |
|----------|-------|
| **F1-Score** | **54.36%** |
| **Exact Match (EM)** | **32.80%** |

### Interprétation

- **F1-Score de 54.36%** : Le modèle trouve une correspondance partielle correcte dans plus de la moitié des cas
- **Exact Match de 32.80%** : Le modèle trouve la réponse exacte dans environ 1/3 des cas
- Ces résultats sont dans la moyenne pour un modèle QA en arabe sur des données réelles

### Comparaison avec l'état de l'art

| Modèle | F1-Score (TyDi QA Arabic) |
|--------|---------------------------|
| mBERT | ~50% |
| XLM-RoBERTa | ~55% |
| **Notre modèle** | **54.36%** |
| AraELECTRA | ~60% |

---

## 6. 🚀 Modèle Fine-tuné

### Publication sur Hugging Face Hub

| Élément | Valeur |
|---------|--------|
| **Repository** | `sonomikane/arabert-qa-arabic-wikipedia` |
| **URL** | https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia |
| **Tâche** | Question Answering |
| **Langue** | Arabe (ar) |

### Utilisation

```python
from transformers import pipeline

# Charger le modèle
qa_pipeline = pipeline(
    "question-answering",
    model="sonomikane/arabert-qa-arabic-wikipedia",
    tokenizer="sonomikane/arabert-qa-arabic-wikipedia"
)

# Poser une question
result = qa_pipeline(
    question="ما هي عاصمة مصر؟",
    context="مصر دولة عربية تقع في شمال أفريقيا. عاصمتها القاهرة."
)

print(result)
# {'answer': 'القاهرة', 'score': 0.95, 'start': 42, 'end': 49}
```

---

## 7. 🌐 Application Streamlit

### Fonctionnalités

1. **Recherche automatique** dans Wikipedia arabe (composant RETRIEVAL)
2. **Extraction de réponses** avec le modèle fine-tuné (composant GENERATION)
3. **Interface bilingue** arabe/français
4. **Affichage des sources** Wikipedia
5. **Score de confiance** pour chaque réponse

### Déploiement

| Plateforme | URL |
|------------|-----|
| **GitHub** | https://github.com/mysonomikane/arabic-qa-streamlit |
| **Streamlit Cloud** | Déploiement automatique depuis GitHub |

### Technologies Utilisées

| Technologie | Version | Usage |
|-------------|---------|-------|
| Streamlit | 1.29.0 | Interface web |
| Transformers | 4.36.0 | Modèle QA |
| PyTorch | Latest | Backend ML |
| Requests | Latest | API Wikipedia |

### Architecture de l'Application

```
streamlit_app/
├── app.py              # Application principale
├── requirements.txt    # Dépendances
└── .streamlit/
    └── config.toml     # Configuration Streamlit
```

---

## 8. 📁 Structure du Projet

```
nlp/
├── projet_NLP_WIKI.ipynb      # Notebook d'entraînement (Colab)
├── test_model.py              # Script de test
├── RAPPORT_PROJET_NLP.md      # Ce rapport
└── streamlit_app/             # Application web
    ├── app.py
    ├── requirements.txt
    └── .streamlit/
        └── config.toml
```

---

## 9. 🔄 Pipeline Complet

### Étape 1 : Préparation des données
- Chargement de TyDi QA, ARCD, XQuAD
- Filtrage des exemples avec réponses
- Tokenization avec AraBERT tokenizer
- Gestion des positions start/end

### Étape 2 : Fine-tuning
- Chargement du modèle AraBERT-v2
- Entraînement sur GPU (Tesla T4)
- 3 epochs avec early stopping
- Sauvegarde du meilleur modèle

### Étape 3 : Évaluation
- Calcul du F1-Score et Exact Match
- Validation sur 3 datasets

### Étape 4 : Publication
- Upload sur Hugging Face Hub
- Documentation du modèle

### Étape 5 : Déploiement
- Création de l'application Streamlit
- Intégration avec Wikipedia API
- Déploiement sur Streamlit Cloud

---

## 10. 💻 Explication Détaillée du Code

### Cellule 1 : Vérification de l'environnement GPU

```python
import torch
if torch.cuda.is_available():
    print(f"GPU DÉTECTÉ : {torch.cuda.get_device_name(0)}")
```

**Objectif :** Vérifier que Google Colab dispose d'un GPU (Tesla T4) pour accélérer l'entraînement. Sans GPU, l'entraînement prendrait plusieurs heures au lieu de ~25 minutes.

---

### Cellule 2 : Installation des dépendances

```python
!pip install -q transformers==4.36.0 datasets==2.16.0
!pip install -q arabert faiss-cpu sentence-transformers
```

**Packages installés :**

| Package | Utilité |
|---------|---------|
| `transformers` | Bibliothèque Hugging Face pour les modèles NLP |
| `datasets` | Chargement des datasets (TyDi QA, ARCD, XQuAD) |
| `arabert` | Préprocesseur spécifique pour le texte arabe |
| `faiss-cpu` | Recherche vectorielle rapide (système RAG) |
| `sentence-transformers` | Création d'embeddings pour la recherche |

---

### Cellule 3 : Chargement des Datasets

```python
# TyDi QA Arabic (~14,000 exemples train)
tydiqa = load_dataset("tydiqa", "primary_task")
tydiqa_train = tydiqa['train'].filter(lambda x: x['language'] == 'arabic')

# Arabic SQuAD / ARCD (~2,500 exemples)
arabic_squad = load_dataset("arcd")

# XQuAD Arabic (validation uniquement)
xquad = load_dataset("xquad", "xquad.ar")
```

**Explication :** On charge trois datasets de Question-Réponse en arabe et on les combine pour avoir plus de données d'entraînement (17,337 exemples au total).

---

### Cellule 4 : Preprocessing des données

#### 4.1. Filtrage des exemples avec réponses

```python
def has_answer_tydiqa(example):
    start_bytes = example['annotations']['minimal_answers_start_byte']
    end_bytes = example['annotations']['minimal_answers_end_byte']
    return start_bytes[0] != -1 and end_bytes[0] != -1
```

**Explication :** On garde uniquement les exemples qui ont une réponse valide (position de début et fin != -1).

#### 4.2. Tokenization avec AraBERT

```python
tokenized = tokenizer(
    questions, 
    contexts, 
    truncation="only_second",  # Tronquer le contexte si trop long
    max_length=384,            # Longueur max en tokens
    stride=128,                # Chevauchement pour les longs textes
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length"
)
```

**Paramètres clés :**
- `max_length=384` : Le modèle BERT a une limite de 512 tokens, on utilise 384 pour la sécurité
- `stride=128` : Si le texte est trop long, on crée des fenêtres qui se chevauchent de 128 tokens
- `return_offsets_mapping` : Permet de mapper les positions de caractères aux positions de tokens

#### 4.3. Calcul des positions start/end

```python
# Conversion bytes → caractères (TyDi QA utilise des bytes)
context_bytes = context.encode('utf-8')
prefix = context_bytes[:start_byte].decode('utf-8')
start_char = len(prefix)
end_char = start_char + len(answer)

# Trouver les tokens correspondants
token_start = context_start
while token_start <= context_end and offsets[token_start][0] <= start_char:
    token_start += 1
token_start -= 1
```

**Explication :** Le modèle QA prédit la position du premier et dernier token de la réponse. On doit donc convertir les positions en caractères vers des positions en tokens.

---

### Cellule 5 : Fine-tuning du modèle

#### 5.1. Chargement du modèle AraBERT

```python
model = AutoModelForQuestionAnswering.from_pretrained("aubmindlab/bert-base-arabertv2")
model = model.cuda()  # Déplacer sur GPU
```

**AraBERT-v2 :** Modèle BERT pré-entraîné sur 77GB de texte arabe (Wikipedia, journaux, livres).

#### 5.2. Configuration de l'entraînement

```python
training_args = TrainingArguments(
    learning_rate=3e-5,      # Taux d'apprentissage standard pour fine-tuning BERT
    num_train_epochs=3,      # 3 passages sur les données
    per_device_train_batch_size=16,
    warmup_ratio=0.1,        # 10% des steps pour augmenter progressivement le LR
    fp16=True,               # Mixed precision pour accélérer sur GPU
    weight_decay=0.01,       # Régularisation L2
)
```

**Justification des hyperparamètres :**

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `learning_rate` | 3e-5 | Valeur standard pour fine-tuning BERT (recommandé: 2e-5 à 5e-5) |
| `epochs` | 3 | Suffisant pour un résultat moyen, évite le surapprentissage |
| `batch_size` | 16 | Compromis mémoire GPU / vitesse |
| `fp16` | True | Divise par 2 l'utilisation mémoire, accélère x2 |
| `warmup_ratio` | 0.1 | Stabilise l'entraînement au début |

#### 5.3. Métriques d'évaluation

```python
def compute_metrics(pred):
    # Exact Match : réponse parfaitement correcte
    em = np.mean((start_pred == start_true) & (end_pred == end_true))
    
    # F1-Score : chevauchement partiel entre prédiction et vérité
    precision = len(pred_tokens & true_tokens) / len(pred_tokens)
    recall = len(pred_tokens & true_tokens) / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
```

**Deux métriques :**
- **Exact Match (EM)** : 1 si la réponse est exactement correcte, 0 sinon
- **F1-Score** : Mesure le chevauchement entre la réponse prédite et la vraie réponse

---

### Cellule 6 : Publication sur Hugging Face Hub

```python
from huggingface_hub import login
login()  # Connexion avec token

model.push_to_hub("sonomikane/arabert-qa-arabic-wikipedia")
tokenizer.push_to_hub("sonomikane/arabert-qa-arabic-wikipedia")
```

**Résultat :** Le modèle est accessible publiquement sur https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia

---

### Cellule 7 : Système RAG avec Wikipedia

#### 7.1. Téléchargement Wikipedia arabe

```python
wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train")
# ~600,000 articles
```

#### 7.2. Création des embeddings

```python
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = encoder.encode(paragraph_texts, batch_size=256)
```

**Sentence-Transformers :** Convertit chaque paragraphe en un vecteur de 384 dimensions qui capture le sens sémantique.

#### 7.3. Index FAISS

```python
index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)
```

**FAISS :** Bibliothèque Facebook pour la recherche rapide de vecteurs similaires. Permet de trouver les paragraphes les plus pertinents en millisecondes.

---

### Cellule 8 : Application Streamlit

```python
def answer_from_wikipedia(question, num_results=3):
    # 1. RETRIEVAL : Chercher les paragraphes pertinents
    query_embedding = encoder.encode([question])
    scores, indices = index.search(query_embedding, num_results)
    
    # 2. Combiner les contextes
    combined_context = " ".join([paragraphs[i]['text'] for i in indices[0]])
    
    # 3. GENERATION : Extraire la réponse avec AraBERT
    result = qa_pipeline(question=question, context=combined_context)
    return result['answer'], result['score']
```

**Architecture RAG :**

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Question   │────▶│  1. RETRIEVAL    │────▶│  Paragraphes    │
│  utilisateur│     │  (FAISS search)  │     │  Wikipedia      │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  Contexte        │
                    │  combiné         │
                    └──────────────────┘
                            │
                            ▼
┌─────────────┐     ┌──────────────────┐
│  Réponse    │◀────│  2. GENERATION   │
│  extraite   │     │  (AraBERT QA)    │
└─────────────┘     └──────────────────┘
```

---

## 11. 🎓 Conclusion

### Objectifs Atteints

✅ Modèle QA fine-tuné sur données arabes  
✅ F1-Score de 54.36% (objectif "résultat moyen" atteint)  
✅ Publication sur Hugging Face Hub  
✅ Application web avec recherche Wikipedia  
✅ Système RAG fonctionnel  

### Améliorations Possibles

1. **Plus d'epochs** : Entraîner sur 5-10 epochs pour de meilleurs résultats
2. **Data Augmentation** : Ajouter plus de données d'entraînement
3. **Modèle plus grand** : Utiliser AraBERT-large au lieu de base
4. **Index local** : Créer un index FAISS de Wikipedia pour une recherche plus rapide
5. **Caching** : Mettre en cache les résultats Wikipedia

---

## 12. 📚 Références

- [AraBERT Paper](https://arxiv.org/abs/2003.00104)
- [TyDi QA Dataset](https://ai.google.com/research/tydiqa)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)

---

## 13. 🔗 Liens du Projet

| Ressource | Lien |
|-----------|------|
| **Modèle HuggingFace** | https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia |
| **Code GitHub** | https://github.com/mysonomikane/arabic-qa-streamlit |
| **Application Live** | Streamlit Cloud (auto-déployée) |
| **AraBERT Original** | https://huggingface.co/aubmindlab/bert-base-arabertv2 |

---

*Rapport généré le 12 Janvier 2026*
