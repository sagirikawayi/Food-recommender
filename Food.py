import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="AI Food Recommender (Scientific Standards)",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# 0.5 自定义 CSS 美化 (含深色模式防护)
# ==========================================
st.markdown("""
    <style>
    /* 核心补丁：强制全局保持浅色视觉，防止系统深色模式干扰 */
    :root { color-scheme: light !important; }
    .stApp { 
        background-color: #F0F2F6 !important; 
        color: #31333F !important;
        forced-color-adjust: none !important; 
    }

    /* 修复下拉框光标问题 */
    div[data-baseweb="select"] { cursor: pointer !important; }
    div[data-baseweb="select"] * { cursor: pointer !important; }
    input[aria-autocomplete="list"] { caret-color: transparent !important; cursor: pointer !important; }

    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb; }
    h1 { color: #1E3A8A !important; font-weight: 800; margin-bottom: 0px; }
    .stSubheader { color: #4B5563 !important; margin-top: -10px; }

    /* 指标卡片悬浮效果 (保留) */
    [data-testid="stMetric"] {
        background-color: #ffffff !important; border: none; padding: 15px 20px;
        border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover { transform: translateY(-5px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }

    /* 按钮美化 (保留原始 2.1.0 蓝色) */
    div.stButton > button:first-child {
        background-color: #3B82F6 !important; color: white !important; width: 100%; border-radius: 10px;
        height: 3.5em; font-weight: bold; border: none; box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.4); margin-top: 20px;
    }

    .stContainer { background-color: #ffffff !important; padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    .stAlert { border: 1px solid #3B82F6 !important; background-color: #ffffff !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 1. 加载数据
# ==========================================
@st.cache_data
def load_and_mine_data():
    try:
        df_nutri = pd.read_csv('healthy_foods_database.csv')
        cols = ['food_name', 'calories', 'protein_g', 'carbs_g', 'sugar_g', 'fat_g', 'sodium_mg']
        return df_nutri[cols].dropna().reset_index(drop=True)
    except:
        st.error("🚨 Dataset Not Found!")
        return pd.DataFrame()


df = load_and_mine_data()
if df.empty: st.stop()

# ==========================================
# 1.5 预计算动态维度 (用于侧边栏显示)
# ==========================================
nutrient_prefixes = ['cal', 'pro', 'carb', 'sugar', 'fat', 'sod']
active_count = sum([st.session_state.get(f"use_{p}", True) for p in nutrient_prefixes])

# ==========================================
# 2. 侧边栏导航逻辑 (完全保留原始版本及所有 Expander)
# ==========================================
with st.sidebar:
    st.title("🧭 Navigation")
    menu_selection = st.selectbox(
        "Choose a module:",
        ["Main Dashboard", "AI Control Panel", "Scientific Standards"]
    )
    st.write("---")

    if menu_selection == "AI Control Panel":
        st.markdown("### ⚙️ AI Control Panel")
        with st.expander("🔍 Algorithm Settings", expanded=True):
            k_val = st.slider("Top-K Matches", 1, 8, 5)
            show_math = st.toggle("🧪 Algorithm X-Ray Vision")

        with st.expander("🏥 Data Health Status", expanded=True):
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.metric("Total Rows", len(df))
                st.metric("Null Cells", df.isnull().sum().sum())
            with h_col2:
                st.metric("Dimensions", f"{active_count}D")
                st.caption("Status: Healthy ✅")

        if st.button("🔄 Reset System Cache"):
            st.cache_data.clear()
            st.rerun()

    elif menu_selection == "Scientific Standards":
        st.markdown("### 📘 Scientific Standards")
        st.info("""
        This AI engine evaluates targets using:
        - 🇬🇧 **FSA Traffic Lights**: Risk assessment.
        - 🇨🇳 **GB 28050**: China's national standards.
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/3034/3034833.png", width=100)
        k_val, show_math = 5, False

    else:
        st.markdown("### 🏠 Welcome")
        st.info("""
        **AI Nutritional Navigator v2.1**

        Define your ideal nutritional signature in the workspace, and let our **KNN engine** find the most scientifically aligned food profiles for you.

        *Switch to **AI Control Panel** to adjust matching sensitivity.*
        """)
        k_val, show_math = 5, False

# ==========================================
# 3. 主界面逻辑
# ==========================================
st.title("🧠 AI Smart Food Recommender")
st.subheader("Scientific Standards-Based Goal Matching")
st.markdown("---")


def sync_val(prefix, source):
    if source == 'slider':
        st.session_state[f'{prefix}_input'] = st.session_state[f'{prefix}_slider']
    else:
        st.session_state[f'{prefix}_slider'] = st.session_state[f'{prefix}_input']


nutrient_map = {'cal': 'calories', 'pro': 'protein_g', 'carb': 'carbs_g', 'sugar': 'sugar_g', 'fat': 'fat_g',
                'sod': 'sodium_mg'}
for prefix, col in nutrient_map.items():
    if f'{prefix}_slider' not in st.session_state:
        st.session_state[f'{prefix}_slider'] = int(df[col].mean())
    if f'{prefix}_input' not in st.session_state:
        st.session_state[f'{prefix}_input'] = st.session_state[f'{prefix}_slider']

st.markdown("### 📋 Step 1: Set Your Nutritional Target")
active_features, user_target_values = [], []


def render_nutrient_control(label, prefix, col_name, emoji):
    # 使用 border=True 确保滑动轴和数字框在同一个明显的框内
    with st.container(border=True):
        is_active = st.checkbox(f"{emoji} Use {label}", value=True, key=f"use_{prefix}")
        c1, c2 = st.columns([3, 1.2])
        min_v, max_v = int(df[col_name].min()), int(df[col_name].max())
        with c1:
            st.slider(label, min_v, max_v, key=f"{prefix}_slider", on_change=sync_val, args=(prefix, 'slider'),
                      disabled=not is_active, label_visibility="collapsed")
        with c2:
            st.number_input(label, min_v, max_v, key=f"{prefix}_input", on_change=sync_val, args=(prefix, 'input'),
                            disabled=not is_active, label_visibility="collapsed")

        if is_active:
            val = st.session_state[f"{prefix}_slider"]
            if prefix == 'pro' and val >= 12.0: st.success("💪 **High Protein**")
            elif prefix == 'fat' and val <= 3.0: st.success("🟢 **Low Fat**")
            elif prefix == 'sugar' and val <= 5.0: st.success("🟢 **Low Sugar**")
            elif prefix == 'sod' and val >= 600: st.error("🔴 **High Sodium**")
            
            active_features.append(col_name)
            user_target_values.append(val)


render_nutrient_control("Calories (kcal)", "cal", "calories", "⚡")
render_nutrient_control("Protein (g)", "pro", "protein_g", "🥚")
render_nutrient_control("Carbs (g)", "carb", "carbs_g", "🍞")
render_nutrient_control("Sugar (g)", "sugar", "sugar_g", "🍭")
render_nutrient_control("Total Fat (g)", "fat", "fat_g", "🥑")
render_nutrient_control("Sodium (mg)", "sod", "sodium_mg", "🧂")

if st.button("🚀 Run AI Search", type="primary"):
    if not active_features:
        st.error("❌ Please select at least one nutrient!")
    else:
        with st.spinner('Calculating...'):
            X = df[active_features].copy()
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            user_target_scaled = scaler.transform([user_target_values])
            knn = NearestNeighbors(n_neighbors=k_val, metric='euclidean').fit(X_scaled)
            distances, indices = knn.kneighbors(user_target_scaled)

            st.markdown("### ✅ Scientific Match Results")
            for i in range(k_val):
                row = df.iloc[indices[0][i]]
                dist = distances[0][i]
                match_score = max(0, (1 - dist) * 100)
                with st.container():
                    st.markdown(f"<h3 style='color: #1E3A8A;'>Rank {i+1}: {row['food_name']}</h3>", unsafe_allow_html=True)
                    cols = st.columns(3)
                    for idx, feat in enumerate(active_features):
                        # 仅修改展示标签，去除冗余后缀
                        clean_name = feat.replace('_g', '').replace('_mg', '').replace('_', ' ').title()
                        unit = " kcal" if 'calories' in feat.lower() else (" mg" if 'sodium' in feat.lower() else " g")
                        cols[idx % 3].metric(clean_name, f"{row[feat]}{unit}")
                    
                    if show_math:
                        st.code(f"Distance: {dist:.4f} | Confidence: {match_score:.1f}%")
                    else:
                        st.progress(int(match_score))
