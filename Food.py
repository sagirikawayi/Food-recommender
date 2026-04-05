import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="AI Food Recommender (Scientific Standards)", page_icon="🧠", layout="centered")
st.title("🧠 AI Smart Food Recommender")
st.subheader("(FSA Traffic Light & GB 28050 Standards)")
st.markdown("---")


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
# 2. Session State 同步逻辑
# ==========================================
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

# ==========================================
# 3. 前端界面：双标准动态面板
# ==========================================
st.subheader("📋 Step 1: Set Your Nutritional Target")
active_features = []
user_target_values = []


def render_nutrient_control(label, prefix, col_name, emoji):
    is_active = st.checkbox(f"{emoji} Use {label}", value=True, key=f"use_{prefix}")
    c1, c2 = st.columns([3, 1.2])
    min_v, max_v = int(df[col_name].min()), int(df[col_name].max())

    with c1:
        st.slider(label, min_v, max_v, key=f"{prefix}_slider", on_change=sync_val, args=(prefix, 'slider'),
                  disabled=not is_active, label_visibility="collapsed")
    with c2:
        st.number_input(label, min_v, max_v, key=f"{prefix}_input", on_change=sync_val, args=(prefix, 'input'),
                        disabled=not is_active, label_visibility="collapsed")

    # --- 科学评判标准逻辑 ---
    if is_active:
        val = st.session_state[f"{prefix}_slider"]

        # A. 第二类：鼓励性成分 (蛋白质 - 中国国标 GB 28050)
        if prefix == 'pro':
            if val >= 12.0:
                st.success(f"💪 **High Protein** (GB 28050: ≥ 12g/100g)")
            elif val >= 6.0:
                st.info(f"✅ **Source of Protein** (Standard: 6g - 12g)")
            else:
                st.warning(f"⚠️ **Low Protein Content** (Below High-Pro Line)")

        # B. 第一类：限制性成分 (脂肪、糖、钠 - 英国 FSA 红绿灯)
        elif prefix == 'fat':
            if val <= 3.0:
                st.success(f"🟢 **Low Fat** (FSA: ≤ 3.0g)")
            elif val <= 17.5:
                st.warning(f"🟡 **Medium Fat** (FSA: 3.0g - 17.5g)")
            else:
                st.error(f"🔴 **High Fat** (FSA: > 17.5g)")

        elif prefix == 'sugar':
            if val <= 5.0:
                st.success(f"🟢 **Low Sugar** (FSA: ≤ 5.0g)")
            elif val <= 22.5:
                st.warning(f"🟡 **Medium Sugar** (FSA: 5.0g - 22.5g)")
            else:
                st.error(f"🔴 **High Sugar** (FSA: > 22.5g)")

        elif prefix == 'sod':
            if val <= 120:
                st.success(f"🟢 **Low Sodium** (FSA: ≤ 120mg)")
            elif val <= 600:
                st.warning(f"🟡 **Medium Sodium** (FSA: 120mg - 600mg)")
            else:
                st.error(f"🔴 **High Sodium** (FSA: > 600mg)")

        active_features.append(col_name)
        user_target_values.append(val)
    st.markdown("---")


# 依次渲染 6 个维度
render_nutrient_control("Calories (kcal)", "cal", "calories", "⚡")
render_nutrient_control("Protein (g)", "pro", "protein_g", "🥚")
render_nutrient_control("Carbs (g)", "carb", "carbs_g", "🍞")
render_nutrient_control("Sugar (g)", "sugar", "sugar_g", "🍭")
render_nutrient_control("Total Fat (g)", "fat", "fat_g", "🥑")
render_nutrient_control("Sodium (mg)", "sod", "sodium_mg", "🧂")

# 算法透视眼开关
show_math = st.toggle("🔍 Activate 'Algorithm X-Ray Vision'")

# ==========================================
# 4. 执行 AI 搜索
# ==========================================
if st.button("🚀 Run AI Search", type="primary"):
    if not active_features:
        st.error("❌ Please select at least one nutrient!")
    else:
        with st.spinner('Matching food profiles...'):
            X = df[active_features].copy()
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            user_target_scaled = scaler.transform([user_target_values])

            k = min(5, len(df))
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(X_scaled)
            distances, indices = knn.kneighbors(user_target_scaled)

            st.success(f"✅ Scientific Match Results (FSA & GB 28050):")
            for i in range(k):
                orig_idx = indices[0][i]
                row = df.iloc[orig_idx]
                dist = distances[0][i]
                match_score = max(0, (1 - dist) * 100)

                with st.container():
                    st.markdown(f"### 🏆 Rank {i + 1}: {row['food_name']}")
                    cols = st.columns(3)

                    for idx, feat in enumerate(active_features):
                        # 1. 标题美化：去除下划线、去除单位后缀、首字母大写
                        clean_title = feat.replace('_g', '').replace('_mg', '').replace('_Mg', '').replace('_',
                                                                                                           ' ').title()

                        # 2. 单位动态判定
                        if 'calories' in feat.lower():
                            u = " kcal"
                        elif 'sodium' in feat.lower():
                            u = " mg"
                        else:
                            u = " g"

                        # 3. 渲染指标
                        cols[idx % 3].metric(clean_title, f"{row[feat]}{u}")

                    # 算法透视眼逻辑保持不变...
                    if show_math:
                        st.info(f"**Euclidean Distance:** `{dist:.4f}` | **Match Confidence:** `{match_score:.1f}%`")
                    else:
                        st.progress(int(match_score))
                    st.markdown("---")
