import streamlit as st
from transformers import pipeline
import requests

# Configuration de la page
st.set_page_config(
    page_title="نظام الإجابة على الأسئلة",
    page_icon="🤖",
    layout="centered"
)

# CSS pour le support RTL (arabe)
st.markdown("""
<style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .stTextInput input {
        direction: rtl;
        text-align: right;
        font-size: 18px;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        direction: rtl;
        text-align: right;
        margin: 20px 0;
    }
    .answer-box h2 {
        margin: 0;
        font-size: 24px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
        margin-top: 10px;
        font-size: 14px;
    }
    .confidence {
        background-color: rgba(255,255,255,0.2);
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === Charger le modèle depuis Hugging Face ===
@st.cache_resource
def load_model():
    """Charge le modèle QA fine-tuné depuis Hugging Face"""
    return pipeline(
        "question-answering",
        model="sonomikane/arabert-qa-arabic-wikipedia",
        tokenizer="sonomikane/arabert-qa-arabic-wikipedia",
        device=-1  # CPU
    )

# === Recherche Wikipedia ===
def search_wikipedia(query, num_results=3):
    """Recherche dans Wikipedia arabe et retourne le contenu"""
    try:
        # Rechercher des articles
        search_url = "https://ar.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json"
        }
        response = requests.get(search_url, params=search_params, timeout=10)
        search_results = response.json()
        
        if "query" not in search_results or not search_results["query"]["search"]:
            return None, []
        
        # Récupérer le contenu des articles
        titles = [r["title"] for r in search_results["query"]["search"]]
        content_params = {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "extracts",
            "exintro": False,
            "explaintext": True,
            "exlimit": num_results,
            "format": "json"
        }
        response = requests.get(search_url, params=content_params, timeout=10)
        content_results = response.json()
        
        # Combiner le contenu
        pages = content_results.get("query", {}).get("pages", {})
        contexts = []
        sources = []
        
        for page_id, page in pages.items():
            if page_id != "-1" and "extract" in page:
                text = page["extract"][:2000]  # Limiter la taille
                if text:
                    contexts.append(text)
                    sources.append(page.get("title", ""))
        
        combined_context = " ".join(contexts)
        return combined_context, sources
        
    except Exception as e:
        return None, []

# === Interface principale ===
st.markdown("""
# 🤖 اسألني أي سؤال بالعربية
## Posez-moi une question en arabe

أكتب سؤالك وسأبحث في ويكيبيديا العربية لأجد الإجابة
""")

st.divider()

# Charger le modèle
with st.spinner("جاري تحميل النموذج... (Chargement du modèle...)"):
    qa_pipeline = load_model()

# Zone de question
question = st.text_input(
    "❓ سؤالك (Votre question)",
    placeholder="مثال: ما هي عاصمة مصر؟",
    key="question_input"
)

# Bouton de recherche
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍 ابحث عن الإجابة", type="primary", use_container_width=True)

# Exemples de questions
st.markdown("### 💡 أمثلة على الأسئلة:")
example_cols = st.columns(3)
examples = [
    "ما هي عاصمة مصر؟",
    "من هو أحمد شوقي؟", 
    "ما هو نهر النيل؟"
]

for i, ex in enumerate(examples):
    with example_cols[i]:
        if st.button(f"📌 {ex}", key=f"ex_{i}", use_container_width=True):
            st.session_state.selected_question = ex

# Utiliser l'exemple sélectionné
if "selected_question" in st.session_state:
    question = st.session_state.selected_question
    del st.session_state.selected_question
    search_button = True

# === Traitement de la question ===
if search_button and question:
    with st.spinner("🔍 جاري البحث في ويكيبيديا..."):
        # Rechercher dans Wikipedia
        context, sources = search_wikipedia(question)
        
        if context:
            with st.spinner("🤔 جاري تحليل الإجابة..."):
                try:
                    # Utiliser le modèle QA
                    result = qa_pipeline(
                        question=question,
                        context=context,
                        max_answer_len=150
                    )
                    
                    answer = result["answer"]
                    score = result["score"]
                    
                    # Afficher la réponse
                    st.markdown(f"""
                    <div class="answer-box">
                        <h2>📝 {answer}</h2>
                        <div class="confidence">🎯 الثقة: {score*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barre de confiance
                    st.progress(score)
                    
                    # Sources
                    if sources:
                        st.markdown("**📚 المصادر (Sources):**")
                        for src in sources:
                            st.markdown(f"- [{src}](https://ar.wikipedia.org/wiki/{src.replace(' ', '_')})")
                    
                    # Contexte utilisé (optionnel)
                    with st.expander("📄 عرض النص المستخدم (Voir le contexte)"):
                        st.markdown(f'<div class="source-box">{context[:1500]}...</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ خطأ في التحليل: {str(e)}")
        else:
            st.warning("⚠️ لم أجد معلومات كافية. جرب سؤالاً آخر.")

elif search_button:
    st.warning("⚠️ الرجاء كتابة سؤال")

# === Footer ===
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 14px;">
    🤖 <strong>Modèle:</strong> <a href="https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia">sonomikane/arabert-qa-arabic-wikipedia</a><br>
    📊 Fine-tuné sur TyDi QA + ARCD + XQuAD | F1: 54.36%<br>
    🔍 Recherche automatique dans Wikipedia Arabe
</div>
""", unsafe_allow_html=True)
