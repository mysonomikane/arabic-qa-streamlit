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

## 10. 🎓 Conclusion

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

## 11. 📚 Références

- [AraBERT Paper](https://arxiv.org/abs/2003.00104)
- [TyDi QA Dataset](https://ai.google.com/research/tydiqa)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)

---

## 12. 🔗 Liens du Projet

| Ressource | Lien |
|-----------|------|
| **Modèle HuggingFace** | https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia |
| **Code GitHub** | https://github.com/mysonomikane/arabic-qa-streamlit |
| **Application Live** | Streamlit Cloud (auto-déployée) |
| **AraBERT Original** | https://huggingface.co/aubmindlab/bert-base-arabertv2 |

---

*Rapport généré le 11 Janvier 2026*
