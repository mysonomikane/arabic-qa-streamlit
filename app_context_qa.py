import streamlit as st
from transformers import pipeline

# Configuration de la page
st.set_page_config(
    page_title="نظام الإجابة على الأسئلة العربية",
    page_icon="🤖",
    layout="wide"
)

# CSS pour le support RTL (arabe)
st.markdown("""
<style>
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
        font-size: 18px;
        font-family: 'Amiri', 'Traditional Arabic', serif;
    }
    .stTextInput input {
        direction: rtl;
        text-align: right;
        font-size: 20px;
        padding: 15px;
        font-family: 'Amiri', 'Traditional Arabic', serif;
    }
    .answer-box {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        direction: rtl;
        text-align: right;
        margin: 20px 0;
        font-size: 24px;
        font-family: 'Amiri', 'Traditional Arabic', serif;
    }
    .context-display {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
        margin: 10px 0;
        border-right: 5px solid #1e88e5;
        font-family: 'Amiri', 'Traditional Arabic', serif;
        font-size: 16px;
        line-height: 1.8;
    }
    .score-box {
        background-color: #e8f5e9;
        padding: 10px 20px;
        border-radius: 25px;
        color: #2e7d32;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    .header-arabic {
        direction: rtl;
        text-align: center;
        font-family: 'Amiri', 'Traditional Arabic', serif;
    }
    .info-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# === Charger le modèle depuis Hugging Face ===
@st.cache_resource
def load_model():
    """Charge le modèle QA fine-tuné depuis Hugging Face"""
    with st.spinner("⏳ جاري تحميل النموذج... (قد يستغرق دقيقة واحدة)"):
        return pipeline(
            "question-answering",
            model="sonomikane/arabert-qa-arabic-wikipedia",
            tokenizer="sonomikane/arabert-qa-arabic-wikipedia",
            device=-1  # CPU, utiliser 0 pour GPU
        )

# === Interface principale ===
def main():
    # Titre
    st.markdown("<h1 class='header-arabic'>🤖 نظام الإجابة على الأسئلة العربية</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-arabic' style='font-size: 18px; color: #666;'>أدخل النص والسؤال وسيجد النموذج الإجابة</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charger le modèle
    try:
        qa_model = load_model()
        st.success("✅ تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
        return
    
    # Créer deux colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 النص (السياق)")
        
        # Exemple de contexte
        example_context = """الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب عادةً الذكاء البشري. تشمل هذه المهام التعلم والاستدلال وحل المشكلات والتعرف على الأنماط ومعالجة اللغة الطبيعية. يعتبر آلان تورينج من أوائل العلماء الذين ساهموا في تطوير هذا المجال من خلال اقتراح اختبار تورينج عام 1950. اليوم، يُستخدم الذكاء الاصطناعي في العديد من التطبيقات مثل المساعدين الافتراضيين والسيارات ذاتية القيادة والتشخيص الطبي."""
        
        context = st.text_area(
            "أدخل النص الذي يحتوي على المعلومات:",
            value=example_context,
            height=250,
            placeholder="اكتب أو الصق النص هنا...",
            help="هذا هو النص الذي سيبحث فيه النموذج عن الإجابة"
        )
        
        st.markdown("### ❓ السؤال")
        
        question = st.text_input(
            "اكتب سؤالك:",
            value="من اقترح اختبار تورينج؟",
            placeholder="مثال: ما هو الذكاء الاصطناعي؟",
            help="اطرح سؤالاً يمكن الإجابة عليه من النص أعلاه"
        )
    
    with col2:
        st.markdown("### 💡 نصائح")
        st.markdown("""
        <div class='info-box'>
        <b>للحصول على أفضل النتائج:</b><br><br>
        ✅ تأكد من أن الإجابة موجودة في النص<br><br>
        ✅ اطرح أسئلة واضحة ومحددة<br><br>
        ✅ استخدم أسئلة تبدأ بـ: من، ما، أين، متى، كيف<br><br>
        ✅ النص يجب أن يكون باللغة العربية
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 معلومات النموذج")
        st.info("""
        **النموذج:** AraBERT v2 Large
        
        **المعلمات:** 355 مليون
        
        **التدريب:** TyDi QA Arabic
        
        **المطور:** sonomikane
        """)
    
    st.markdown("---")
    
    # Bouton de recherche
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        search_button = st.button("🔍 ابحث عن الإجابة", type="primary", use_container_width=True)
    
    # Traitement de la question
    if search_button:
        if not context.strip():
            st.warning("⚠️ يرجى إدخال النص أولاً")
            return
        
        if not question.strip():
            st.warning("⚠️ يرجى إدخال السؤال")
            return
        
        with st.spinner("🔍 جاري البحث عن الإجابة..."):
            try:
                # Obtenir la réponse du modèle
                result = qa_model(
                    question=question,
                    context=context,
                    max_answer_len=100,
                    handle_impossible_answer=True
                )
                
                answer = result['answer']
                score = result['score']
                
                st.markdown("---")
                st.markdown("## 📌 النتيجة")
                
                # Afficher la réponse
                if answer and score > 0.01:
                    st.markdown(f"""
                    <div class='answer-box'>
                        <h3 style='margin-bottom: 15px;'>💡 الإجابة:</h3>
                        <p style='font-size: 28px; margin: 0;'>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Score de confiance
                    confidence_pct = score * 100
                    if confidence_pct >= 70:
                        confidence_color = "#2e7d32"
                        confidence_text = "ثقة عالية"
                        confidence_emoji = "🟢"
                    elif confidence_pct >= 40:
                        confidence_color = "#f57c00"
                        confidence_text = "ثقة متوسطة"
                        confidence_emoji = "🟡"
                    else:
                        confidence_color = "#d32f2f"
                        confidence_text = "ثقة منخفضة"
                        confidence_emoji = "🔴"
                    
                    st.markdown(f"""
                    <div style='text-align: center; margin-top: 20px;'>
                        <span style='background-color: {confidence_color}20; color: {confidence_color}; 
                                     padding: 10px 25px; border-radius: 25px; font-size: 18px;'>
                            {confidence_emoji} {confidence_text}: {confidence_pct:.1f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Afficher le contexte avec la réponse surlignée
                    with st.expander("📖 عرض النص مع تمييز الإجابة"):
                        highlighted_context = context.replace(
                            answer, 
                            f"<mark style='background-color: #ffeb3b; padding: 2px 5px;'>{answer}</mark>"
                        )
                        st.markdown(f"<div class='context-display'>{highlighted_context}</div>", unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ لم يتمكن النموذج من إيجاد إجابة واضحة في النص المقدم.")
                    st.info("💡 حاول إعادة صياغة السؤال أو تأكد من أن المعلومات موجودة في النص.")
                    
            except Exception as e:
                st.error(f"❌ حدث خطأ: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🔬 تم تطوير هذا النظام باستخدام AraBERT و Hugging Face Transformers</p>
        <p>📚 المشروع: نظام الإجابة على الأسئلة باللغة العربية</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
