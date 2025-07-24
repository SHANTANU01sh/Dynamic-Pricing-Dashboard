import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import io
import os
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# LangChain and Groq imports
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Professional CSS with animations and modern theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --primary-dark: #4f46e5;
    --secondary-color: #f1f5f9;
    --accent-color: #10b981;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Main container styling */
.main .block-container {
    padding: 2rem 1rem;
    max-width: 1200px;
}

/* Animated background gradient */
body {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Professional navbar */
.navbar {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: slideDown 0.6s ease-out;
}

@keyframes slideDown {
    from {
        transform: translateY(-100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.navbar button {
    color: var(--text-primary);
    padding: 0.75rem 1.25rem;
    text-decoration: none;
    font-weight: 500;
    border-radius: 12px;
    margin-right: 0.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
    background: none;
    cursor: pointer;
}

.navbar button:hover {
    background: var(--primary-color);
    color: white;
    transform: translateY(-2px);
    box-shadow: var(--shadow-md); /* Corrected from box post-shadow */
}

.navbar button:active {
    transform: translateY(0);
}

.navbar .right {
    float: right;
}

/* Enhanced cards */
.metric-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeInUp 0.6s ease-out;
    position: relative;
    overflow: hidden;
}

@keyframes fadeInUp {
    from {
        transform: translateY(30px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    border-radius: 20px 20px 0 0;
}

/* Professional buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 12px;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
}

.stButton > button:active {
    transform: translateY(0);
}

/* Animated loading spinner */
.loading-spinner {
    border: 3px solid rgba(99, 102, 241, 0.3);
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    width: 24px;
    height: 24px;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced form inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div > select {
    border-radius: 12px;
    border: 2px solid var(--border-color);
    padding: 0.75rem 1rem;
    transition: all 0.3s ease;
    background: rgba(255, 255, 255, 0.95);
    color: #000000;
}

.stTextInput > div > div > input::placeholder,
.stSelectbox > div > div > select::placeholder {
    color: #64748b;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    transform: translateY(-1px);
}

/* Professional titles */
h1, h2, h3 {
    color: var(--text-primary);
    font-weight: 600;
    animation: fadeInDown 0.6s ease-out;
}

@keyframes fadeInDown {
    from {
        transform: translateY(-20px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

/* Success/Error messages */
.stSuccess, .stError, .stInfo {
    border-radius: 12px;
    animation: slideInRight 0.5s ease-out;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Chat messages */
.chat-message {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    animation: messageAppear 0.4s ease-out;
    border-left: 4px solid var(--primary-color);
}

@keyframes messageAppear {
    from {
        transform: scale(0.8);
        opacity: 0;
    }
    to {
        transform: scale(1);
        opacity: 1;
    }
}

/* Data tables */
.dataframe {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-md);
    animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Login/Signup forms */
.auth-container {
    max-width: 400px;
    margin: 2rem auto;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: scaleIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes scaleIn {
    from {
        transform: scale(0.9);
        opacity: 0;
    }
    to {
        transform: scale(1);
        opacity: 1;
    }
}

/* Dashboard stats */
.stat-box {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 0.5rem;
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
    animation: bounceIn 0.6s ease-out;
}

@keyframes bounceIn {
    0% {
        transform: scale(0.3);
        opacity: 0;
    }
    50% {
        transform: scale(1.05);
    }
    70% {
        transform: scale(0.9);
    }
    100% {
        transform: scale(1);
        opacity: 1;
    }
}

.stat-box:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

/* Pulse animation for important elements */
.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
    100% {
        transform: scale(1);
    }
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# DB setup
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    message TEXT,
    response TEXT
)
''')
conn.commit()

# LangChain/Groq Setup
if "groq_chat_model" not in st.session_state:
    try:
        st.session_state.groq_chat_model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    except Exception as e:
        st.error(f"Failed to initialize Groq model. Make sure GROQ_API_KEY is set correctly. Error: {e}")
        st.session_state.groq_chat_model = None

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)

# Prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI Assistant for a dynamic pricing dashboard.
            You can answer questions about product prices, quantities, sales, revenue, and profit margins.
            Provide concise and helpful answers based on typical business data.
            If the question is outside these topics, politely redirect the user to ask about business metrics."""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Utils
def create_user(username, password):
    password_hash = generate_password_hash(password)
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?,?)', (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def check_user(username, password):
    c.execute('SELECT id, password_hash FROM users WHERE username=?', (username,))
    user = c.fetchone()
    if user and check_password_hash(user[1], password):
        return user[0]
    return None

def save_chat(user_id, msg, response):
    c.execute('INSERT INTO chat_history (user_id, message, response) VALUES (?,?,?)', (user_id, msg, response))
    conn.commit()

def get_history(user_id):
    c.execute('SELECT message, response FROM chat_history WHERE user_id=?', (user_id,))
    return c.fetchall()

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'login'

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Update session state page based on query params only on initial load
if 'init_page_set' not in st.session_state:
    query_page = st.query_params.get('page')
    if query_page in ['overview', 'chatbot', 'history', 'logout', 'login', 'signup']:
        st.session_state.page = query_page
    st.session_state.init_page_set = True

# Professional navbar with animations
def show_navbar():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        # Using a consistent key for each button
        if st.button("📊 Overview", key="nav_overview"):
            st.session_state.page = 'overview'
            st.rerun()
    with col2:
        if st.button("🤖 Ask Me", key="nav_chatbot"):
            st.session_state.page = 'chatbot'
            st.rerun()
    with col3:
        if st.button("📜 History", key="nav_history"):
            st.session_state.page = 'history'
            st.rerun()
    with col4:
        if st.button("🚪 Logout", key="nav_logout"):
            st.session_state.page = 'logout'
            st.rerun() # Force rerun to trigger logout_page logic
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced login page
def login_page():
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--primary-color); font-size: 2.5rem; margin-bottom: 0.5rem;">🔐</h1>
        <h2 style="margin-bottom: 0.5rem;">Welcome Back</h2>
        <p style="color: var(--text-secondary);">Sign in to your account</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username", key="login_username")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Login", use_container_width=True, key="login_button_main"):
            if username and password:
                user_id = check_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.page = 'overview'
                    st.success("Login successful! 🎉")
                    st.rerun() # Force rerun to navigate to dashboard
                else:
                    st.error("Invalid credentials ❌")
            else:
                st.warning("Please fill in all fields")

    with col2:
        if st.button("📝 Sign Up", use_container_width=True, key="signup_button_redirect"):
            st.session_state.page = 'signup'
            st.rerun() # Force rerun to navigate to signup

    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced signup page
def signup_page():
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--accent-color); font-size: 2.5rem; margin-bottom: 0.5rem;">📝</h1>
        <h2 style="margin-bottom: 0.5rem;">Create Account</h2>
        <p style="color: var(--text-secondary);">Join us today</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Choose Username", placeholder="Enter a unique username", key="signup_username")
    password = st.text_input("Choose Password", type="password", placeholder="Create a strong password", key="signup_password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Create Account", use_container_width=True, key="create_account_button"):
            if username and password:
                if len(password) >= 6:
                    if create_user(username, password):
                        st.success("Account created successfully! 🎉")
                        st.session_state.page = 'login'
                        st.balloons()
                        st.rerun() # Force rerun to navigate to login
                    else:
                        st.error("Username already exists ❌")
                else:
                    st.warning("Password must be at least 6 characters")
            else:
                st.warning("Please fill in all fields")

    with col2:
        if st.button("🔙 Back to Login", use_container_width=True, key="back_to_login_button"):
            st.session_state.page = 'login'
            st.rerun() # Force rerun to navigate to login

    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced overview page with professional dashboard
def overview_page():
    show_navbar()

    # Hero section
    st.markdown("""
    <div class="metric-card pulse" style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--primary-color); font-size: 3rem; margin-bottom: 0.5rem;">🧪 Dynamic Pricing Dashboard</h1>
        <p style="font-size: 1.2rem; color: var(--text-secondary);">Advanced ML-powered pricing analytics</p>
        <p style="color: var(--text-secondary);">Powered by Python • MLflow • Prefect</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample data (since we don't have the actual file)
    @st.cache_data
    def load_data():
        # Generate sample data for demonstration
        np.random.seed(42)
        n_products = 100

        skus = [f"SKU-{1000+i}" for i in range(n_products)]
        quantities = np.random.randint(10, 1000, n_products)
        prices = np.random.uniform(100, 2000, n_products)
        sales = quantities * prices
        costs = prices * np.random.uniform(0.5, 0.8, n_products)

        return pd.DataFrame({
            'sku': skus,
            'total_quantity': quantities,
            'suggested_price': prices,
            'total_sales': sales,
            'cost': costs
        })

    df = load_data()

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3 style="color: var(--primary-color); margin: 0;">📦 {len(df):,}</h3>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">Total Products</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_revenue = df['total_sales'].sum()
        st.markdown(f"""
        <div class="stat-box">
            <h3 style="color: var(--accent-color); margin: 0;">💰 ₹{total_revenue:,.0f}</h3>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">Total Revenue</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_price = df['suggested_price'].mean()
        st.markdown(f"""
        <div class="stat-box">
            <h3 style="color: var(--primary-color); margin: 0;">📈 ₹{avg_price:.0f}</h3>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">Avg Price</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        df['margin'] = ((df['suggested_price'] - df['cost']) / df['suggested_price'])
        avg_margin = df['margin'].mean()
        st.markdown(f"""
        <div class="stat-box">
            <h3 style="color: var(--accent-color); margin: 0;">📊 {avg_margin:.1%}</h3>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">Avg Margin</p>
        </div>
        """, unsafe_allow_html=True)

    # Filters section
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🔍 Filters & Controls")

    col1, col2 = st.columns(2)
    with col1:
        sku_filter = st.text_input("🔎 Filter SKU contains:", placeholder="Enter SKU to search...", key="sku_filter")
        if sku_filter:
            df = df[df['sku'].str.contains(sku_filter, case=False, na=False)]

    with col2:
        margin_range = st.slider("📊 Margin Range", 0.0, 1.0, (0.2, 0.6), format="%.1f", key="margin_range")
        df = df[(df['margin'] >= margin_range[0]) & (df['margin'] <= margin_range[1])]

    st.markdown('</div>', unsafe_allow_html=True)

    # Top products section
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🏆 Top-Selling Products")
    top10 = df.sort_values('total_quantity', ascending=False).head(10)

    # Display as cards instead of table
    for i in range(0, len(top10), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(top10):
                product = top10.iloc[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.7);
                                border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
                                border-left: 4px solid var(--primary-color);">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-color);">{product['sku']}</h4>
                        <p style="margin: 0; color: var(--text-secondary);">Qty: {product['total_quantity']:,} |
                        Price: ₹{product['suggested_price']:.0f} |
                        Margin: {product['margin']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

    st.bar_chart(top10.set_index('sku')['total_quantity'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts section
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📈 Price Distribution")
    st.bar_chart(df.set_index('sku')['suggested_price'].head(20))
    st.markdown('</div>', unsafe_allow_html=True)

    # Data upload section
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload New Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], help="Upload sales data to refresh predictions", key="data_uploader")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Predictions", key="refresh_button"):
            with st.spinner("Processing..."):
                import time
                time.sleep(2)  # Simulate processing
            st.success("Predictions refreshed successfully! ✨")

    with col2:
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Export section
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("⬇ Export Data")

    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download CSV",
            csv_data,
            file_name='suggested_prices.csv',
            mime='text/csv',
            use_container_width=True,
            key="download_csv_button"
        )

    with col2:
        excel_buf = io.BytesIO()
        df.to_excel(excel_buf, index=False)
        st.download_button(
            "📊 Download Excel",
            excel_buf.getvalue(),
            file_name='suggested_prices.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True,
            key="download_excel_button"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem;
                background: rgba(255, 255, 255, 0.7); border-radius: 12px;">
        <p style="color: var(--text-secondary);">Made with ❤ using Streamlit • MLflow • Prefect</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced chatbot page
def chatbot_page():
    show_navbar()

    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <h1 style="color: var(--primary-color);">🤖 AI Assistant</h1>
        <p style="color: var(--text-secondary);">Ask me anything about your business data</p>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history from session state
    for msg in st.session_state.chat_memory.buffer:
        if isinstance(msg, HumanMessage):
            st.markdown(f"""
            <div class="chat-message">
                <strong style="color: var(--primary-color);">You:</strong> {msg.content}
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"""
            <div class="chat-message" style="border-left-color: var(--accent-color);">
                <strong style="color: var(--accent-color);">AI Assistant:</strong> {msg.content}
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("💬 Your message:", key="chat_input")

    if user_input:
        if st.session_state.groq_chat_model:
            with st.spinner("Thinking..."):
                # Append user message to memory
                st.session_state.chat_memory.chat_memory.add_user_message(user_input)

                # Create the chain
                chain = prompt | st.session_state.groq_chat_model

                # Invoke the model
                response = chain.invoke({
                    "input": user_input,
                    "history": st.session_state.chat_memory.buffer
                }).content

                # Append AI response to memory
                st.session_state.chat_memory.chat_memory.add_ai_message(response)

                # Save to database
                save_chat(st.session_state.user_id, user_input, response)

                # Rerun to display updated chat history
                st.rerun()
        else:
            st.error("AI Chatbot not initialized. Please check your Groq API key.")

# Enhanced history page
def history_page():
    show_navbar()

    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <h1 style="color: var(--primary-color);">📜 Chat History</h1>
        <p style="color: var(--text-secondary);">Your conversation history with AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    history = get_history(st.session_state.user_id)

    if history:
        for i, (message, response) in enumerate(reversed(history)):
            st.markdown(f"""
            <div class="chat-message" style="animation-delay: {i * 0.1}s;">
                <strong style="color: var(--primary-color);">You:</strong> {message}
            </div>
            <div class="chat-message" style="border-left-color: var(--accent-color); animation-delay: {i * 0.1 + 0.05}s;">
                <strong style="color: var(--accent-color);">AI Assistant:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3>📭 No chat history yet</h3>
            <p style="color: var(--text-secondary);">Start chatting with the AI assistant to see your history here!</p>
        </div>
        """, unsafe_allow_html=True)
        # Add a Streamlit button to navigate to chat, as HTML onclick is not ideal
        if st.button("🤖 Go to Chat →", key="go_to_chat_from_history"):
            st.session_state.page = 'chatbot'
            st.rerun()


def logout_page():
    st.session_state.user_id = None
    st.session_state.page = 'login'
    st.session_state.chat_memory.clear()  # Clear chat memory on logout
    st.rerun()

# Enhanced router with page transitions
current_page = st.session_state.page

if st.session_state.user_id:
    if current_page == 'overview':
        overview_page()
    elif current_page == 'chatbot':
        chatbot_page()
    elif current_page == 'history':
        history_page()
    elif current_page == 'logout':
        logout_page() # This function already handles st.rerun()
    else:
        # Default authenticated page if current_page is not recognized
        st.session_state.page = 'overview'
        overview_page()
else:
    if current_page == 'login':
        login_page()
    elif current_page == 'signup':
        signup_page()
    else:
        # Default unauthenticated page if current_page is not recognized
        st.session_state.page = 'login'
        login_page()
