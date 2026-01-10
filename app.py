
import streamlit as st
from transformers import pipeline
import torch

# Configuration de la page
st.set_page_config(
    page_title="نظام الإجابة على الأسئلة",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le support RTL (arabe)
st.markdown("""
<style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }
    .stTextInput input {
        direction: rtl;
        text-align: right;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
    }
    .source-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# === Charger le modèle (cache pour performance) ===
@st.cache_resource
def load_model():
    """Charge le modèle QA une seule fois"""
    return pipeline(
        "question-answering",
        model="aubmindlab/bert-base-arabertv2",  # Ou votre modèle HuggingFace
        tokenizer="aubmindlab/bert-base-arabertv2",
        device=-1  # CPU pour Streamlit Cloud
    )

# Charger le modèle
with st.spinner("جاري تحميل النموذج... (Loading model...)"):
    qa_pipeline = load_model()

# === Interface principale ===
st.markdown("""
# 🔍 نظام الإجابة على الأسئلة بالعربية
## Arabic Question Answering System

اكتب نصاً من ويكيبيديا العربية ثم اسأل سؤالاً عنه
""")

st.divider()

# === Colonnes pour l'interface ===
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de texte pour le contexte
    context = st.text_area(
        "📄 النص (Context)",
        placeholder="الصق هنا نصاً من ويكيبيديا العربية...",
        height=200,
        help="Collez ici un paragraphe de Wikipedia arabe"
    )
    
    # Zone pour la question
    question = st.text_input(
        "❓ السؤال (Question)",
        placeholder="اكتب سؤالك هنا...",
        help="Écrivez votre question en arabe"
    )
    
    # Bouton de soumission
    submit = st.button("🔍 ابحث عن الإجابة", type="primary", use_container_width=True)

with col2:
    st.markdown("### 💡 أمثلة")
    
    # Exemples prédéfinis
    examples = {
        "عاصمة مصر": {
            "question": "ما هي عاصمة مصر؟",
            "context": "مصر دولة عربية تقع في شمال أفريقيا. عاصمتها القاهرة وهي أكبر مدينة في العالم العربي."
        },
        "جامعة القاهرة": {
            "question": "متى تأسست الجامعة؟",
            "context": "تأسست جامعة القاهرة في عام 1908 وهي من أقدم الجامعات في مصر والوطن العربي."
        },
        "نهر النيل": {
            "question": "ما هو طول نهر النيل؟",
            "context": "نهر النيل هو أطول أنهار العالم، يبلغ طوله حوالي 6650 كيلومتر."
        }
    }
    
    for name, data in examples.items():
        if st.button(f"📌 {name}", use_container_width=True):
            st.session_state.example_q = data["question"]
            st.session_state.example_c = data["context"]
            st.rerun()

# Utiliser l'exemple si sélectionné
if "example_q" in st.session_state:
    question = st.session_state.example_q
    context = st.session_state.example_c
    del st.session_state.example_q
    del st.session_state.example_c

# === Traitement de la question ===
if submit and question and context:
    with st.spinner("جاري البحث عن الإجابة..."):
        try:
            result = qa_pipeline(
                question=question,
                context=context,
                max_answer_len=100
            )
            
            answer = result["answer"]
            score = result["score"]
            
            st.divider()
            
            # Afficher la réponse
            st.markdown("### 💡 الإجابة (Answer)")
            st.markdown(f"""
            <div class="answer-box">
                <h3>📝 {answer}</h3>
                <p>🎯 الثقة: {score*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barre de progression pour la confiance
            st.progress(score)
            
        except Exception as e:
            st.error(f"❌ خطأ: {str(e)}")

elif submit:
    st.warning("⚠️ الرجاء إدخال النص والسؤال")

# === Sidebar avec infos ===
with st.sidebar:
    st.markdown("## 📊 معلومات النظام")
    st.markdown("""
    **Modèle:** AraBERT-v2
    
    **Datasets d'entraînement:**
    - TyDi QA Arabic
    - ARCD (Arabic SQuAD)
    - XQuAD Arabic
    
    **Métriques:**
    - F1-Score: 54.36%
    - Exact Match: 32.80%
    """)
    
    st.divider()
    
    st.markdown("### 🔗 Liens utiles")
    st.markdown("""
    - [Wikipedia Arabe](https://ar.wikipedia.org)
    - [AraBERT](https://huggingface.co/aubmindlab)
    - [Code source](https://github.com)
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    Made with ❤️ using Streamlit & AraBERT
</div>
""", unsafe_allow_html=True)
