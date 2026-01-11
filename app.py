import streamlit as st
from transformers import pipeline
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration de la page
st.set_page_config(
    page_title="مساعد ويكيبيديا العربية",
    page_icon="🔍",
    layout="centered"
)

# CSS pour le support RTL (arabe)
st.markdown("""
<style>
    .stTextInput input {
        direction: rtl;
        text-align: right;
        font-size: 20px;
        padding: 15px;
    }
    .answer-box {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        direction: rtl;
        text-align: right;
        margin: 20px 0;
    }
    .context-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
        margin: 10px 0;
        border-left: 4px solid #1e88e5;
    }
    .source-link {
        background-color: #e3f2fd;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
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

# === Session requêtes robuste pour Wikipedia ===
def get_wikipedia_session():
    """Crée une session robuste pour accéder à Wikipedia API"""
    session = requests.Session()
    
    # Stratégie de retry robuste
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Headers complets
    session.headers.update({
        "User-Agent": "ArabicQABot/1.0 (Arabic Wikipedia QA; +https://github.com/mysonomikane/arabic-qa-streamlit)",
        "Accept-Language": "ar,en-US;q=0.9",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate"
    })
    
    return session

# === Recherche Wikipedia Arabe ===
def search_wikipedia_arabic(query, num_results=5):
    """
    Recherche dans Wikipedia arabe et retourne le contenu des articles pertinents.
    C'est le composant RETRIEVAL du système RAG.
    """
    try:
        session = get_wikipedia_session()
        api_url = "https://ar.wikipedia.org/w/api.php"
        
        # Étape 1: Rechercher les articles pertinents
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json"
        }
        
        response = session.get(api_url, params=search_params, timeout=20)
        response.raise_for_status()
        search_data = response.json()
        
        if "query" not in search_data or not search_data["query"]["search"]:
            return None, [], "لم يتم العثور على نتائج"
        
        # Récupérer les titres des articles trouvés
        titles = [result["title"] for result in search_data["query"]["search"]]
        
        # Délai pour respecter le rate limiting de Wikipedia
        time.sleep(0.5)
        
        # Étape 2: Récupérer le contenu des articles
        content_params = {
            "action": "query",
            "titles": "|".join(titles[:3]),  # Limiter à 3 articles
            "prop": "extracts",
            "exintro": False,
            "explaintext": True,
            "exlimit": 3,
            "format": "json"
        }
        
        response = session.get(api_url, params=content_params, timeout=20)
        response.raise_for_status()
        content_data = response.json()
        
        # Extraire le contenu
        pages = content_data.get("query", {}).get("pages", {})
        contexts = []
        sources = []
        
        for page_id, page in pages.items():
            if page_id != "-1" and "extract" in page:
                text = page["extract"]
                # Prendre les premiers 1500 caractères de chaque article
                if len(text) > 100:
                    contexts.append(text[:1500])
                    sources.append({
                        "title": page.get("title", ""),
                        "url": f"https://ar.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}"
                    })
        
        if not contexts:
            return None, [], "لم يتم العثور على محتوى"
        
        # Combiner tous les contextes
        combined_context = "\n\n".join(contexts)
        return combined_context, sources, None
        
    except requests.exceptions.Timeout:
        return None, [], "⏱️ انتهت مهلة الاتصال بويكيبيديا (Timeout). حاول لاحقاً."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            return None, [], "🚫 رفضت ويكيبيديا الطلب (403). جرب سؤالاً آخر."
        elif e.response.status_code == 429:
            return None, [], "⏳ طلبات كثيرة. انتظر قليلاً ثم حاول مجدداً."
        else:
            return None, [], f"❌ خطأ HTTP: {e.response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, [], "❌ خطأ في الاتصال بويكيبيديا. تحقق من اتصالك بالإنترنت."
    except Exception as e:
        return None, [], f"❌ خطأ: {str(e)}"

# === Interface principale ===
st.markdown("""
# 🔍 مساعد ويكيبيديا العربية
## Arabic Wikipedia Assistant

**اسألني أي سؤال وسأبحث في ويكيبيديا العربية لأجد الإجابة!**

🤖 هذا النظام يستخدم نموذج AraBERT المُدرَّب على بيانات ويكيبيديا العربية
""")

st.divider()

# Charger le modèle
with st.spinner("⏳ جاري تحميل النموذج... (Chargement du modèle)"):
    qa_pipeline = load_model()

# Zone de question principale
question = st.text_input(
    "❓ اكتب سؤالك هنا:",
    placeholder="مثال: ما هي عاصمة مصر؟ من هو طه حسين؟ متى تأسست جامعة القاهرة؟",
    key="main_question"
)

# Exemples de questions
st.markdown("### 💡 أمثلة على الأسئلة:")
col1, col2, col3 = st.columns(3)

example_questions = [
    "ما هي عاصمة مصر؟",
    "من هو نجيب محفوظ؟",
    "ما هو نهر النيل؟",
    "متى تأسست جامعة القاهرة؟",
    "من هو صلاح الدين الأيوبي؟",
    "ما هي الأهرامات؟"
]

cols = st.columns(3)
for i, q in enumerate(example_questions):
    with cols[i % 3]:
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.main_question = q
            st.rerun()

st.divider()

# Bouton de recherche
search_clicked = st.button(
    "🔍 ابحث في ويكيبيديا",
    type="primary",
    use_container_width=True
)

# === Traitement de la question ===
if search_clicked and question:
    
    # Étape 1: Recherche dans Wikipedia (RETRIEVAL)
    with st.spinner("🔍 جاري البحث في ويكيبيديا العربية..."):
        context, sources, error = search_wikipedia_arabic(question)
    
    if error:
        st.error(f"❌ {error}")
    elif context:
        # Afficher les sources trouvées
        st.success(f"✅ تم العثور على {len(sources)} مقالات ذات صلة")
        
        # Étape 2: Extraction de la réponse (GENERATION)
        with st.spinner("🤔 جاري تحليل المعلومات واستخراج الإجابة..."):
            try:
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
                    <p style="font-size: 16px; opacity: 0.9;">❓ السؤال: {question}</p>
                    <h2 style="font-size: 28px; margin: 15px 0;">📝 {answer}</h2>
                    <p style="font-size: 14px;">🎯 نسبة الثقة: {score*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Barre de confiance
                st.progress(score)
                
                # Avertissement si confiance faible
                if score < 0.3:
                    st.warning("⚠️ نسبة الثقة منخفضة. قد تحتاج لإعادة صياغة السؤال.")
                
                # Afficher les sources
                st.markdown("### 📚 المصادر من ويكيبيديا:")
                for src in sources:
                    st.markdown(f'<span class="source-link">📄 <a href="{src["url"]}" target="_blank">{src["title"]}</a></span>', unsafe_allow_html=True)
                
                # Afficher le contexte utilisé
                with st.expander("📖 عرض النص المستخدم من ويكيبيديا"):
                    st.markdown(f'<div class="context-box">{context[:2000]}...</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ خطأ في التحليل: {str(e)}")
    else:
        st.warning("⚠️ لم أجد معلومات كافية. جرب سؤالاً آخر أو أعد صياغته.")

elif search_clicked:
    st.warning("⚠️ الرجاء كتابة سؤال أولاً")

# === Sidebar avec informations ===
with st.sidebar:
    st.markdown("## ℹ️ عن النظام")
    st.markdown("""
    **🤖 كيف يعمل النظام (RAG):**
    
    1️⃣ **البحث (Retrieval)**
    - يبحث في ويكيبيديا العربية
    - يجلب المقالات ذات الصلة
    
    2️⃣ **الاستخراج (Generation)**
    - يحلل النصوص المُسترجعة
    - يستخرج الإجابة الدقيقة
    
    ---
    
    **📊 معلومات النموذج:**
    - **الاسم:** AraBERT-QA
    - **التدريب:** TyDi QA + ARCD + XQuAD
    - **F1-Score:** 54.36%
    - **Exact Match:** 32.80%
    
    ---
    
    **🔗 الروابط:**
    """)
    
    st.markdown("[🤗 النموذج على HuggingFace](https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia)")
    st.markdown("[📖 ويكيبيديا العربية](https://ar.wikipedia.org)")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    🔍 <strong>Arabic Wikipedia QA Assistant</strong> | 
    نظام RAG للإجابة على الأسئلة من ويكيبيديا العربية<br>
    <a href="https://huggingface.co/sonomikane/arabert-qa-arabic-wikipedia">sonomikane/arabert-qa-arabic-wikipedia</a>
</div>
""", unsafe_allow_html=True)
