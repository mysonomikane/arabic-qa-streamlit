import streamlit as st
from transformers import pipeline

# Configuration de la page
st.set_page_config(
    page_title="نظام الإجابة على الأسئلة",
    page_icon="🤖",
    layout="centered"
)

# CSS pour le support RTL (arabe)
st.markdown("""
<style>
    .stTextInput input, .stTextArea textarea {
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
    .context-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
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
        device=-1
    )

# === Interface principale ===
st.markdown("""
# 🤖 نظام الإجابة على الأسئلة بالعربية
## Arabic Question Answering System

**كيف يعمل النظام:**
1. الصق نصاً من ويكيبيديا أو أي مصدر آخر
2. اكتب سؤالك
3. النظام سيستخرج الإجابة من النص
""")

# Info box
st.markdown("""
<div class="info-box">
⚠️ <strong>ملاحظة مهمة:</strong> هذا النموذج يستخرج الإجابة من النص المقدم. يجب أن تكون الإجابة موجودة في النص.
</div>
""", unsafe_allow_html=True)

st.divider()

# Charger le modèle
with st.spinner("جاري تحميل النموذج... (Chargement du modèle ~1-2 min)"):
    qa_pipeline = load_model()
    st.success("✅ تم تحميل النموذج بنجاح!")

# === Exemples pré-définis ===
st.markdown("### 💡 اختر مثالاً أو أدخل نصك:")

examples = {
    "🇪🇬 مصر": {
        "context": "مصر أو رسمياً جمهورية مصر العربية، دولة عربية تقع في الركن الشمالي الشرقي من قارة أفريقيا. عاصمتها القاهرة وهي أكبر مدينة في العالم العربي وأفريقيا. يبلغ عدد سكان مصر حوالي 104 مليون نسمة. تمتلك مصر سواحل طويلة على البحر الأبيض المتوسط والبحر الأحمر.",
        "questions": ["ما هي عاصمة مصر؟", "كم عدد سكان مصر؟", "أين تقع مصر؟"]
    },
    "🏛️ جامعة القاهرة": {
        "context": "جامعة القاهرة هي ثاني أقدم الجامعات المصرية. تأسست كلياتها المختلفة في عام 1908 وكانت تسمى الجامعة المصرية. تقع في مدينة الجيزة. تضم الجامعة حوالي 200000 طالب وطالبة.",
        "questions": ["متى تأسست جامعة القاهرة؟", "أين تقع الجامعة؟", "كم عدد الطلاب؟"]
    },
    "🌊 نهر النيل": {
        "context": "نهر النيل هو أطول أنهار الكرة الأرضية. يبلغ طوله حوالي 6650 كيلومتر. ينبع النيل من بحيرة فيكتوريا ويصب في البحر الأبيض المتوسط. يمر النيل بعشر دول أفريقية.",
        "questions": ["ما هو طول نهر النيل؟", "من أين ينبع النيل؟", "أين يصب النيل؟"]
    },
    "📝 نص مخصص": {
        "context": "",
        "questions": []
    }
}

# Sélection d'exemple
selected_example = st.selectbox(
    "اختر موضوعاً:",
    list(examples.keys()),
    index=0
)

# Zone de contexte
if selected_example == "📝 نص مخصص":
    context = st.text_area(
        "📄 الصق النص هنا:",
        placeholder="الصق هنا نصاً من ويكيبيديا العربية أو أي مصدر آخر...",
        height=200
    )
    question = st.text_input(
        "❓ اكتب سؤالك:",
        placeholder="مثال: ما هي عاصمة مصر؟"
    )
else:
    context = examples[selected_example]["context"]
    st.markdown(f'<div class="context-box"><strong>📄 النص:</strong><br>{context}</div>', unsafe_allow_html=True)
    
    # Questions suggérées
    st.markdown("**❓ أسئلة مقترحة:**")
    cols = st.columns(len(examples[selected_example]["questions"]))
    question = ""
    
    for i, q in enumerate(examples[selected_example]["questions"]):
        with cols[i]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.selected_q = q
    
    # Question personnalisée ou sélectionnée
    if "selected_q" in st.session_state:
        question = st.session_state.selected_q
        del st.session_state.selected_q
    else:
        question = st.text_input("أو اكتب سؤالك الخاص:", placeholder="...")

# Bouton de recherche
if st.button("🔍 ابحث عن الإجابة", type="primary", use_container_width=True):
    if question and context:
        with st.spinner("🤔 جاري تحليل النص..."):
            try:
                result = qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=100
                )
                
                answer = result["answer"]
                score = result["score"]
                
                # Afficher la réponse
                st.markdown(f"""
                <div class="answer-box">
                    <p>❓ السؤال: {question}</p>
                    <h2>📝 الإجابة: {answer}</h2>
                    <p>🎯 نسبة الثقة: {score*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Barre de confiance
                st.progress(score)
                
                if score < 0.3:
                    st.warning("⚠️ نسبة الثقة منخفضة. قد لا تكون الإجابة دقيقة.")
                    
            except Exception as e:
                st.error(f"❌ خطأ: {str(e)}")
    else:
        st.warning("⚠️ الرجاء إدخال النص والسؤال")

# === Footer ===
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 14px;">
    🤖 <strong>Modèle:</strong> <a href="https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia">sonomikane/arabert-qa-arabic-wikipedia</a><br>
    📊 Fine-tuné sur TyDi QA + ARCD + XQuAD | F1: 54.36%
</div>
""", unsafe_allow_html=True)
