import streamlit as st
import os
from dotenv import load_dotenv
from chatbot_v03 import HybridKnowledgeBase, TyphoonChatbot
import time
import base64

# Load environment variables
load_dotenv()

# Load background image and encode to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get base64 encoded images
bg_image = get_base64_image("gb.jpg")
bot_logo = get_base64_image("bot_4712206.png")

# Page configuration
st.set_page_config(
    page_title="น้องบอท - ผู้ช่วยหลักสูตรคณะครุศาสตร์อุตสาหกรรม",
    page_icon="bot_4712206.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Modern Green & White Theme with Kanit Font
st.markdown(f"""
<style>
    /* Import Kanit Font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

    /* Global Font Setting */
    * {{
        font-family: 'Kanit', sans-serif !important;
    }}

    /* พื้นหลังหน้าหลัก - Image Background */
    .stApp {{
        background-image: url('data:image/jpeg;base64,{bg_image}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }}

    /* พื้นหลัง Main content area */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {{
        display: none !important;
    }}

    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}

    /* Hide sidebar collapse button */
    button[kind="header"] {{
        display: none !important;
    }}

    /* Header Styling with Shadow */
    .main-header {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(5, 150, 105, 0.1);
        letter-spacing: -0.5px;
    }}

    .sub-header {{
        font-size: 1.3rem;
        font-weight: 600;
        shadow: 1px 1px 3px rgba(4, 120, 87, 0.1);
        color: #047857;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.3px;
    }}

    /* Chat Message Containers with Modern Shadow */
    .chat-message {{
        padding: 1.75rem;
        border-radius: 1rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-direction: column;
        color: #1F2937;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    .chat-message:hover {{
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        transform: translateY(-2px);
    }}

    /* User Message - Clean White Design */
    .user-message {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-left: 5px solid #10B981;
        border: 2px solid #D1FAE5;
        color: #1F2937;
    }}

    /* Bot Message - Soft Green Background */
    .bot-message {{
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        border-left: 5px solid #059669;
        border: 1px solid #D1FAE5;
        color: #1F2937;
    }}

    /* Message Labels */
    .message-label {{
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 1rem;
        letter-spacing: 0.3px;
    }}

    .user-label {{
        color: #10B981;
    }}

    .bot-label {{
        color: #059669;
    }}

    /* Context Box */
    .context-box {{
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 4px solid #10B981;
        margin-top: 0.75rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }}

    /* Stat Cards - Modern with Gradient */
    .stat-card {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        padding: 1.25rem;
        border-radius: 1rem;
        border: 2px solid #10B981;
        text-align: center;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.15);
        transition: all 0.3s ease;
    }}

    .stat-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(16, 185, 129, 0.25);
    }}

    .stat-number {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .stat-label {{
        font-size: 0.95rem;
        font-weight: 400;
        color: #047857;
        margin-top: 0.5rem;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #059669 !important;
        font-weight: 600 !important;
        padding: 0.5rem 0;
        letter-spacing: 0.3px;
    }}

    /* Slider Styling - Modern Green */
    .stSlider > div > div > div > div {{
        background: linear-gradient(90deg, #10B981 0%, #059669 100%) !important;
    }}

    .stSlider > div > div > div {{
        background-color: #D1FAE5 !important;
    }}

    /* Button Styling - Modern with Gradient */
    .stButton > button {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.85rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3) !important;
        letter-spacing: 0.5px;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(5, 150, 105, 0.4) !important;
    }}

    .stButton > button:active {{
        transform: translateY(-1px);
    }}

    /* Checkbox Styling */
    .stCheckbox {{
        padding: 0.75rem 0;
    }}

    .stCheckbox > label {{
        font-weight: 500 !important;
        font-size: 1rem !important;
        color: #047857 !important;
    }}

    /* Info/Success/Warning boxes */
    .stAlert {{
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%) !important;
        border: 2px solid #10B981 !important;
        border-radius: 0.75rem !important;
        color: #047857 !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15) !important;
    }}

    /* Divider Styling */
    [data-testid="stSidebar"] hr {{
        border-color: #D1FAE5 !important;
        border-width: 2px !important;
        margin: 1.5rem 0 !important;
        opacity: 0.6;
    }}

    /* Section Headers with Modern Background */
    [data-testid="stSidebar"] h3 {{
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        padding: 1rem 1.25rem !important;
        border-radius: 0.75rem;
        margin-bottom: 1.25rem !important;
        border-left: 5px solid #10B981;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }}

    /* Expander Styling - Modern Design */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%) !important;
        border-radius: 0.75rem !important;
        border: 2px solid #D1FAE5 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #047857 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }}

    .streamlit-expanderHeader:hover {{
        background: linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 100%) !important;
        border-color: #10B981 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15) !important;
    }}

    /* Chat Input Styling - Modern with Shadow */
    .stChatInput > div {{
        border: 3px solid #10B981 !important;
        border-radius: 1rem !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
        transition: all 0.3s ease !important;
    }}

    .stChatInput > div:focus-within {{
        border-color: #059669 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
        transform: translateY(-2px);
    }}

    .stChatInput input {{
        font-size: 1.05rem !important;
        font-weight: 400 !important;
    }}

    /* Markdown text in sidebar */
    [data-testid="stSidebar"] p {{
        color: #1F2937;
        line-height: 1.8;
        font-weight: 400;
    }}

    [data-testid="stSidebar"] strong {{
        color: #047857;
        font-weight: 600;
    }}

    /* Help text */
    .stSlider [data-testid="stMarkdownContainer"] p {{
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 400;
    }}

    /* Main content h3 styling */
    .main h3 {{
        color: #059669 !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: 0.3px;
    }}

    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: #F0FDF4;
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def init_chatbot():
    """Initialize the chatbot (cached to avoid reloading)"""
    api_key = os.getenv('TYPHOON_API_KEY')
    if not api_key:
        st.error("❌ ไม่พบ TYPHOON_API_KEY ใน environment variables")
        st.stop()

    with st.spinner('🔧 กำลังโหลดระบบ Chatbot...'):
        kb = HybridKnowledgeBase(
            persist_directory="./chroma_db",
            collection_name="chatbot_knowledge",
            use_reranker=True
        )
        chatbot = TyphoonChatbot(api_key, kb, use_compression=False)

    return chatbot

# Initialize chatbot
chatbot = init_chatbot()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_context' not in st.session_state:
    st.session_state.show_context = False

if 'last_contexts' not in st.session_state:
    st.session_state.last_contexts = []

# Default search settings (since sidebar is removed)
n_results = 10
bm25_weight = 0.4
vector_weight = 0.6
show_context = False

# Header
st.markdown(f'<div class="main-header"><img src="data:image/png;base64,{bot_logo}" style="width: 80px; height: 80px; vertical-align: middle; margin-right: 15px;"> น้องบอท</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ผู้ช่วยให้คำปรึกษาหลักสูตร คณะครุศาสตร์อุตสาหกรรม มจพ.</div>', unsafe_allow_html=True)

# Main chat area
st.markdown("### 💬 พื้นที่แชท")

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <div class="message-label user-label">🙋 คุณ:</div>
            <div>{message["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-message bot-message">
            <div class="message-label bot-label"><img src="data:image/png;base64,{bot_logo}" style="width: 28px; height: 28px; vertical-align: middle; margin-right: 8px;"> น้องบอท:</div>
            <div>{message["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)

        # Show context if enabled and available
        if st.session_state.show_context and i < len(st.session_state.last_contexts):
            contexts = st.session_state.last_contexts[i]
            if contexts:
                with st.expander(f"📚 เอกสารอ้างอิง ({len(contexts)} รายการ)", expanded=False):
                    for ctx in contexts:
                        score_info = f"🎯 Score: {ctx.get('score', 0):.3f}"
                        if 'rerank_score' in ctx:
                            score_info += f" | Rerank: {ctx.get('rerank_score', 0):.3f}"

                        st.markdown(f"""
                        **รายการที่ {ctx.get('rank', 0)}** | {score_info}

                        {ctx.get('text', '')}
                        """)
                        st.markdown("---")

# Suggested questions section
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%); padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #D1FAE5; margin-bottom: 1.5rem;">
    <h4 style="color: #047857; margin-bottom: 1rem; font-size: 1.1rem;">💡 ตัวอย่างคำถามที่คุณสามารถถามได้:</h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 0.75rem;">
        <div style="background-color: #FFFFFF; padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #10B981; color: #1F2937; font-size: 0.9rem;">
            📚 หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี
        </div>
        <div style="background-color: #FFFFFF; padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #10B981; color: #1F2937; font-size: 0.9rem;">
            📝 คุณสมบัติการสมัครเรียนมีอะไรบ้าง
        </div>
        <div style="background-color: #FFFFFF; padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #10B981; color: #1F2937; font-size: 0.9rem;">
            💼 จบแล้วทำงานอะไรได้บ้าง
        </div>
        <div style="background-color: #FFFFFF; padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #10B981; color: #1F2937; font-size: 0.9rem;">
            🎓 มีวิชาอะไรบ้างในปี 1
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat input with enhanced styling
user_input = st.chat_input("✨ พิมพ์คำถามของคุณที่นี่... (กด Enter เพื่อส่ง)")

# Footer - Moved below chat input
st.markdown("---")

# Footer content with clear button and contact info
col1, col2 = st.columns([1, 2])

with col1:
    # Clear history button
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        st.session_state.messages = []
        chatbot.clear_history()
        st.session_state.last_contexts = []
        st.rerun()

with col2:
    # Contact box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%); padding: 1.25rem; border-radius: 0.75rem; border: 2px solid #10B981; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);">
        <strong style="color: #047857; font-size: 1.05rem;">📞 ติดต่อสอบถาม</strong><br><br>
        <span style="color: #1F2937; line-height: 1.8;">
        📧 <a href="http://admission.kmutnb.ac.th" style="color: #059669; text-decoration: none; font-weight: 500;">admission.kmutnb.ac.th</a><br>
        ☎️ <span style="font-weight: 500;">02-555-2000</span><br>
        📘 คณะครุศาสตร์อุตสาหกรรม มจพ.
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Copyright
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1.5rem 0; border-top: 2px solid #D1FAE5; margin-top: 2rem;">
    <strong style="color: #047857;">พัฒนาโดย คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ</strong><br>
    <span style="font-size: 0.85rem;">© 2025 KMUTNB. All rights reserved.</span>
</div>
""", unsafe_allow_html=True)

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get expanded query for search
    expanded_query = chatbot.expand_query(user_input)

    # Search for relevant knowledge
    with st.spinner('🔍 กำลังค้นหาข้อมูล...'):
        relevant_knowledge = chatbot.knowledge_base.search_knowledge(
            expanded_query,
            n_results=n_results,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )

    # Store contexts for this response
    st.session_state.last_contexts.append(relevant_knowledge)

    # Generate response
    with st.spinner('🤔 กำลังคิดคำตอบ...'):
        response = chatbot.chat(user_input)

    # Add bot response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun to display new messages
    st.rerun()
