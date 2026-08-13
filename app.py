import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Post-Craniotomy DVT Risk Predictor — XGBoost",
    page_icon="🧠",
    layout="centered",
)

# Light medical / Stripe-ish theme via custom CSS
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .title-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
        color: white; padding: 1.1rem 1.4rem; border-radius: 14px;
        box-shadow: 0 8px 24px rgba(37,99,235,0.25); margin-bottom: 1.2rem;
    }
    .title-banner h1 { color: white; margin: 0; font-size: 1.5rem; }
    .title-banner p  { color: #e0f2fe; margin: 0.3rem 0 0; font-size: 0.9rem; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 0.9rem 1rem; margin-bottom: 0.8rem;
    }
    .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants — MUST match model training
# ----------------------------------------------------------------------------
# Feature order is critical: it must equal model.feature_names_in_
FEATURE_NAMES = ["ETT_24H", "POSITION", "AGE", "BMI", "OP_DUR", "FDP", "CRP"]

# Cutoff locked on the INTERNAL VALIDATION set at 90% sensitivity
# (XGBoost, 7-variable model). Decision rule: high risk if prob >= CUTOFF.
CUTOFF = 0.07275406271219254

# POSITION encoding: numeric, as used in training (POSITION_AS_CATEGORY=False)
POSITION_MAP = {"Left Lateral (左侧卧位)": 0, "Right Lateral (右侧卧位)": 1, "Supine (仰卧位)": 2}
POSITION_LIST = list(POSITION_MAP.keys())

# Global SHAP mean|SHAP| importance (7-variable XGBoost, validation set)
SHAP_GLOBAL = {
    "AGE": 0.9090, "ETT_24H": 0.4002, "BMI": 0.3097, "FDP": 0.3002,
    "OP_DUR": 0.2708, "POSITION": 0.2695, "CRP": 0.2579,
}

MODEL_LABEL = "XGBoost (gradient-boosted trees)"
REFERENCE_LINE = (
    "Wang H, et al. Early Prediction of Postoperative Deep Vein Thrombosis "
    "After Brain Tumor Craniotomy Using an XGBoost Model: Development and "
    "Multi-Center External Validation. (Under review)."
)


# ----------------------------------------------------------------------------
# Model + explainer loading (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("XGBoost_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_explainer(model):
    try:
        import shap
        return shap.TreeExplainer(model)
    except Exception:
        return None


model = load_model()
explainer = load_explainer(model)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="title-banner">
        <h1>🧠 颅脑肿瘤开颅术后 DVT 风险预测工具</h1>
        <p>Post-Craniotomy DVT Risk Predictor · {MODEL_LABEL} · 7-variable model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    本工具基于多中心数据开发的 **XGBoost** 模型，输入 7 项临床变量即可预测
    开颅术后早期下肢深静脉血栓（DVT）风险，并给出是否可安全免除超声筛查的建议。
    """
)

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
st.sidebar.header("📋 患者参数 Patient Parameters")

ett_24h = st.sidebar.radio(
    "术后 24h 内气管插管 / 机械通气 (ETT_24H)", options=["No", "Yes"], index=0,
    help="Endotracheal intubation or mechanical ventilation within 24h postoperatively",
)
age = st.sidebar.number_input(
    "年龄 Age (years)", min_value=18, max_value=90, value=50, step=1
)
bmi = st.sidebar.number_input(
    "体重指数 BMI (kg/m²)", min_value=12.0, max_value=50.0, value=24.0,
    step=0.1, format="%.1f"
)
op_dur = st.sidebar.number_input(
    "手术时长 Operative Duration (hours)", min_value=0.5, max_value=20.0,
    value=4.5, step=0.1, format="%.1f"
)
position = st.sidebar.selectbox(
    "手术体位 Surgical Position", options=POSITION_LIST, index=2
)
fdp = st.sidebar.number_input(
    "纤维蛋白 / 纤维蛋白原降解产物 FDP (μg/mL)", min_value=0.0, max_value=200.0,
    value=5.0, step=0.1, format="%.1f"
)
crp = st.sidebar.number_input(
    "C 反应蛋白 CRP (mg/L)", min_value=0.0, max_value=500.0,
    value=25.0, step=0.1, format="%.1f"
)

# Encode inputs
ett_code = 1 if ett_24h == "Yes" else 0
position_code = POSITION_MAP[position]
features = np.array([[ett_code, position_code, age, bmi, op_dur, fdp, crp]])
input_df = pd.DataFrame(features, columns=FEATURE_NAMES)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if st.sidebar.button("🔍 预测风险 Predict Risk", type="primary"):
    with st.spinner("计算中 Calculating risk..."):
        prob = float(model.predict_proba(input_df)[0, 1])
        pred_class = 1 if prob >= CUTOFF else 0

    st.subheader("📊 预测结果 Prediction Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DVT 概率", f"{prob:.1%}")
    with col2:
        label = "🔴 高风险" if pred_class == 1 else "🟢 低风险"
        st.metric("风险分级", label)
    with col3:
        st.metric("决策阈值", f"{CUTOFF:.3f}")

    st.progress(int(prob * 100))
    st.caption(
        f"预测概率 = {prob:.4f} ｜ 高风险切点 = {CUTOFF:.4f} "
        f"（内部验证集锁定 90% 灵敏度）"
    )

    if pred_class == 1:
        st.warning(
            "⚠️ **高风险 — 预测概率 ≥ 高敏切点**\n\n"
            "**建议：** 加强 DVT 预防措施，并**优先安排**下肢静脉超声筛查。"
        )
    else:
        st.success(
            "✅ **低风险 — 预测概率 < 高敏切点**\n\n"
            "**建议：** 常规术后护理即可，**可安全免除**下肢静脉超声筛查 "
            "（外部验证集中约 32.7% 患者归属此低风险组，且漏诊率极低）。"
        )

    # ---- Per-instance SHAP explanation (graceful fallback) ----
    if explainer is not None:
        try:
            sv = explainer.shap_values(input_df)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.array(sv)[0]
            base = float(explainer.expected_value)
            order = np.argsort(np.abs(sv))[::-1]
            feat_labels = [
                "ETT_24H", "POSITION", "AGE", "BMI", "OP_DUR", "FDP", "CRP"
            ]

            fig, ax = plt.subplots(figsize=(7, 3.6))
            vals = sv[order]
            labels = [feat_labels[i] for i in order]
            colors = ["#dc2626" if v > 0 else "#2563eb" for v in vals]
            y = np.arange(len(labels))
            ax.barh(y, vals, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=10)
            ax.invert_yaxis()
            ax.axvline(0, color="#475569", lw=0.8)
            ax.set_xlabel("SHAP 贡献 (log-odds)", fontsize=9)
            ax.set_title(
                f"单例概率归因 · 基准 log-odds={base:.3f} → 预测概率={prob:.1%}",
                fontsize=10,
            )
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.markdown("**🧩 该患者的风险归因 (SHAP)**")
            st.caption(
                "红 = 推高风险，蓝 = 降低风险。各特征贡献叠加于基准值即为最终预测概率。"
            )
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP 单例解释暂不可用：{e}")
    else:
        st.info(
            "ℹ️ 当前环境未安装 `shap`，已跳过单例归因图。"
            "在 requirements 中安装 shap 即可启用。"
        )

    # Input summary
    with st.expander("📋 查看输入参数 View Input Parameters"):
        st.json({
            "ETT_24H (术后24h插管)": ett_24h,
            "POSITION (体位)": position,
            "AGE (岁)": age,
            "BMI (kg/m²)": bmi,
            "OP_DUR (小时)": op_dur,
            "FDP (μg/mL)": fdp,
            "CRP (mg/L)": crp,
        })


# ----------------------------------------------------------------------------
# Sidebar footer
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    f"**模型:** {MODEL_LABEL}\n\n"
    "**切点:** 0.0728 (内部验证集 90% 灵敏度)\n\n"
    "**外部验证 (n=303, VTE=86):**\n"
    "AUC: 0.764 | 灵敏度: 0.919\n"
    "特异度: 0.424 | NPV: 0.929\n"
    "免超声比例: 32.7%\n\n"
    "© 2026 北京天坛医院 &\n重庆大学附属肿瘤医院"
)


# ----------------------------------------------------------------------------
# Model explanation (global) + performance
# ----------------------------------------------------------------------------
with st.expander("📈 模型说明 · 全局特征重要性 & 外部验证表现"):
    st.markdown("**全局特征重要性（平均 |SHAP| 值，验证集）**")
    imp = (
        pd.DataFrame(
            {"Feature": list(SHAP_GLOBAL.keys()),
             "Mean|SHAP|": list(SHAP_GLOBAL.values())}
        )
        .sort_values("Mean|SHAP|", ascending=True)
    )
    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    ax2.barh(imp["Feature"], imp["Mean|SHAP|"], color="#2563eb")
    ax2.set_xlabel("Mean |SHAP| value", fontsize=9)
    ax2.set_title("Feature Importance (XGBoost, 7-variable)", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("**外部测试集表现（303 例，VTE 86 例）**")
    perf = pd.DataFrame({
        "指标 Metric": ["AUC", "灵敏度 Sensitivity", "特异度 Specificity",
                         "PPV", "NPV", "Accuracy", "免超声比例"],
        "数值 Value": ["0.764 (0.709–0.826)", "0.919 (0.841–0.960)",
                       "0.424 (0.360–0.490)", "0.387", "0.929",
                       "0.564", "32.7% (92/303)"],
    })
    st.table(perf)
    st.caption(
        "阈值在内部验证集锁定为 90% 灵敏度（cutoff≈0.0728），"
        "外部测试集沿用该切点盲评。高灵敏度设计以‘安全免除超声’为目标，"
        "低风险组漏诊率极低。"
    )

# Main page footer
st.markdown("---")
st.markdown(f"*Reference: {REFERENCE_LINE}*")
