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
# Bilingual text
# ----------------------------------------------------------------------------
TEXT = {
    "zh": {
        "lang_label": "语言 / Language",
        "title": "🧠 颅脑肿瘤开颅术后 DVT 风险预测工具",
        "subtitle": "Post-Craniotomy DVT Risk Predictor · XGBoost · 7 变量模型",
        "intro": (
            "本工具基于多中心数据开发的 **XGBoost** 模型，输入 7 项临床变量即可预测"
            "开颅术后早期下肢深静脉血栓（DVT）风险，并给出是否可安全免除超声筛查的建议。"
        ),
        "sidebar_header": "📋 患者参数",
        "ett_label": "术后 24h 内气管插管 / 机械通气 (ETT_24H)",
        "ett_help": "Endotracheal intubation or mechanical ventilation within 24h postoperatively",
        "age": "年龄 Age (岁)",
        "bmi": "体重指数 BMI (kg/m²)",
        "opdur": "手术时长 Operative Duration (小时)",
        "position": "手术体位 Surgical Position",
        "fdp": "纤维蛋白/纤维蛋白原降解产物 FDP (μg/mL)",
        "crp": "C 反应蛋白 CRP (mg/L)",
        "predict_btn": "🔍 预测风险",
        "result_header": "📊 预测结果",
        "prob_label": "DVT 概率",
        "risk_label": "风险分级",
        "high_risk_label": "🔴 高风险",
        "low_risk_label": "🟢 低风险",
        "cutoff_label": "决策阈值",
        "progress_caption": "预测概率 = {p} ｜ 高风险切点 = {c}（内部验证集锁定 90% 灵敏度）",
        "high_risk": (
            "⚠️ **高风险 — 预测概率 ≥ 高敏切点**\n\n"
            "**建议：** 加强 DVT 预防措施，并**优先安排**下肢静脉超声筛查。"
        ),
        "low_risk": (
            "✅ **低风险 — 预测概率 < 高敏切点**\n\n"
            "**建议：** 常规术后护理即可，**可安全免除**下肢静脉超声筛查"
            "（外部验证集中约 32.7% 患者归属此低风险组，且漏诊率极低）。"
        ),
        "shap_heading": "**🧩 该患者的风险归因 (SHAP)**",
        "shap_caption": "红 = 推高风险，蓝 = 降低风险。各特征贡献叠加于基准值即为最终预测概率。",
        "shap_unavailable": "ℹ️ 当前环境未安装 `shap`，已跳过单例归因图。在 requirements 中安装 shap 即可启用。",
        "input_expander": "📋 查看输入参数",
        "k_ett": "ETT_24H (术后24h插管)",
        "k_position": "POSITION (体位)",
        "k_age": "AGE (岁)",
        "k_bmi": "BMI (kg/m²)",
        "k_opdur": "OP_DUR (小时)",
        "k_fdp": "FDP (μg/mL)",
        "k_crp": "CRP (mg/L)",
        "sidebar_footer": (
            "**模型:** {model}\n\n"
            "**切点:** 0.0728 (内部验证集 90% 灵敏度)\n\n"
            "**外部验证 (n=303, VTE=86):**\n"
            "AUC: 0.764 | 灵敏度: 0.919\n"
            "特异度: 0.424 | NPV: 0.929\n"
            "免超声比例: 32.7%\n\n"
            "© 2026 北京天坛医院 & 重庆大学附属肿瘤医院"
        ),
        "model_exp_header": "📈 模型说明 · 全局特征重要性 & 外部验证表现",
        "imp_heading": "**全局特征重要性（平均 |SHAP| 值，验证集）**",
        "perf_heading": "**外部测试集表现（303 例，VTE 86 例）**",
        "perf_metric_col": "指标",
        "perf_value_col": "数值",
        "m_auc": "AUC",
        "m_sens": "灵敏度",
        "m_spec": "特异度",
        "m_ppv": "PPV",
        "m_npv": "NPV",
        "m_acc": "准确率",
        "m_spare": "免超声比例",
        "perf_caption": (
            "阈值在内部验证集锁定为 90% 灵敏度（cutoff≈0.0728），"
            "外部测试集沿用该切点盲评。高灵敏度设计以‘安全免除超声’为目标，低风险组漏诊率极低。"
        ),
        "model_label": "XGBoost（梯度提升树）",
    },
    "en": {
        "lang_label": "语言 / Language",
        "title": "🧠 DVT Risk Predictor After Craniotomy",
        "subtitle": "XGBoost model · 7 variables · multicenter external validation",
        "intro": (
            "An XGBoost model developed on multicenter data. Enter 7 clinical variables to predict "
            "the risk of early postoperative deep vein thrombosis (DVT) after craniotomy, with a "
            "recommendation on whether ultrasound screening can be safely omitted."
        ),
        "sidebar_header": "📋 Patient Parameters",
        "ett_label": "Post-op 24h intubation / mechanical ventilation (ETT_24H)",
        "ett_help": "Endotracheal intubation or mechanical ventilation within 24h postoperatively",
        "age": "Age (years)",
        "bmi": "BMI (kg/m²)",
        "opdur": "Operative duration (hours)",
        "position": "Surgical position",
        "fdp": "Fibrin/fibrinogen degradation products FDP (μg/mL)",
        "crp": "C-reactive protein CRP (mg/L)",
        "predict_btn": "🔍 Predict Risk",
        "result_header": "📊 Prediction Result",
        "prob_label": "DVT Probability",
        "risk_label": "Risk Level",
        "high_risk_label": "🔴 High risk",
        "low_risk_label": "🟢 Low risk",
        "cutoff_label": "Decision cut-off",
        "progress_caption": "Predicted probability = {p} | High-risk cut-off = {c} (locked at 90% sensitivity on internal validation set)",
        "high_risk": (
            "⚠️ **High risk — predicted probability ≥ high-sensitivity cut-off**\n\n"
            "**Recommendation:** reinforce DVT prophylaxis and **prioritize** lower-limb venous "
            "ultrasound screening."
        ),
        "low_risk": (
            "✅ **Low risk — predicted probability < high-sensitivity cut-off**\n\n"
            "**Recommendation:** routine postoperative care is sufficient; lower-limb venous "
            "ultrasound can be **safely omitted** (≈32.7% of external test patients fell in this "
            "low-risk group with minimal missed cases)."
        ),
        "shap_heading": "**🧩 Risk attribution for this patient (SHAP)**",
        "shap_caption": "Red = pushes risk up; blue = pushes risk down. Each feature's contribution added to the baseline yields the final predicted probability.",
        "shap_unavailable": "ℹ️ `shap` is not installed in this environment; the per-patient attribution plot is skipped. Install shap via requirements to enable it.",
        "input_expander": "📋 View Input Parameters",
        "k_ett": "ETT_24H (post-op 24h intubation)",
        "k_position": "POSITION",
        "k_age": "AGE (years)",
        "k_bmi": "BMI (kg/m²)",
        "k_opdur": "OP_DUR (hours)",
        "k_fdp": "FDP (μg/mL)",
        "k_crp": "CRP (mg/L)",
        "sidebar_footer": (
            "**Model:** {model}\n\n"
            "**Cut-off:** 0.0728 (90% sensitivity, internal validation)\n\n"
            "**External test (n=303, VTE=86):**\n"
            "AUC: 0.764 | Sensitivity: 0.919\n"
            "Specificity: 0.424 | NPV: 0.929\n"
            "Ultrasound-sparing: 32.7%\n\n"
            "© 2026 Beijing Tiantan Hospital & The Affiliated Cancer Hospital of Chongqing University"
        ),
        "model_exp_header": "📈 Model Notes · Global Feature Importance & External Validation",
        "imp_heading": "**Global feature importance (mean |SHAP|, validation set)**",
        "perf_heading": "**External test-set performance (303 patients, 86 VTE)**",
        "perf_metric_col": "Metric",
        "perf_value_col": "Value",
        "m_auc": "AUC",
        "m_sens": "Sensitivity",
        "m_spec": "Specificity",
        "m_ppv": "PPV",
        "m_npv": "NPV",
        "m_acc": "Accuracy",
        "m_spare": "Ultrasound-sparing rate",
        "perf_caption": (
            "The cut-off was locked at 90% sensitivity on the internal validation set "
            "(cutoff≈0.0728) and applied unchanged to the external test set. The high-sensitivity "
            "design targets safe ultrasound omission with minimal missed cases in the low-risk group."
        ),
        "model_label": "XGBoost (gradient-boosted trees)",
    },
}

LANG = st.sidebar.selectbox(TEXT["zh"]["lang_label"], ["中文", "English"], index=0)
t = TEXT[LANG]

# ----------------------------------------------------------------------------
# Constants — MUST match model training
# ----------------------------------------------------------------------------
FEATURE_NAMES = ["ETT_24H", "POSITION", "AGE", "BMI", "OP_DUR", "FDP", "CRP"]
CUTOFF = 0.07275406271219254  # locked on internal validation set at 90% sensitivity

POSITION_OPTIONS = ["Left Lateral", "Right Lateral", "Supine"]
POSITION_CODE = {"Left Lateral": 0, "Right Lateral": 1, "Supine": 2}
POSITION_DISPLAY = {
    "zh": ["左侧卧位", "右侧卧位", "仰卧位"],
    "en": POSITION_OPTIONS,
}

# Global SHAP mean|SHAP| importance (7-variable XGBoost, validation set)
SHAP_GLOBAL = {
    "AGE": 0.9090, "ETT_24H": 0.4002, "BMI": 0.3097, "FDP": 0.3002,
    "OP_DUR": 0.2708, "POSITION": 0.2695, "CRP": 0.2579,
}

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


model = load_model()


@st.cache_resource
def load_explainer():
    # Do NOT pass `model` as a cached-function argument: Streamlit hashes the
    # argument and the XGBoost model is unhashable (UnhashableParamError).
    # Reference the module-level global instead.
    try:
        import shap
        return shap.TreeExplainer(model)
    except Exception:
        return None


explainer = load_explainer()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="title-banner">
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(t["intro"])

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
st.sidebar.header(t["sidebar_header"])

ett_24h = st.sidebar.radio(t["ett_label"], options=["No", "Yes"], index=0, help=t["ett_help"])
age = st.sidebar.number_input(t["age"], min_value=18, max_value=90, value=50, step=1)
bmi = st.sidebar.number_input(t["bmi"], min_value=12.0, max_value=50.0, value=24.0, step=0.1, format="%.1f")
op_dur = st.sidebar.number_input(t["opdur"], min_value=0.5, max_value=20.0, value=4.5, step=0.1, format="%.1f")
position = st.sidebar.selectbox(t["position"], options=POSITION_DISPLAY[LANG], index=2)
fdp = st.sidebar.number_input(t["fdp"], min_value=0.0, max_value=200.0, value=5.0, step=0.1, format="%.1f")
crp = st.sidebar.number_input(t["crp"], min_value=0.0, max_value=500.0, value=25.0, step=0.1, format="%.1f")

ett_code = 1 if ett_24h == "Yes" else 0
position_idx = POSITION_DISPLAY[LANG].index(position)
position_code = POSITION_CODE[POSITION_OPTIONS[position_idx]]
features = np.array([[ett_code, position_code, age, bmi, op_dur, fdp, crp]])
input_df = pd.DataFrame(features, columns=FEATURE_NAMES)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if st.sidebar.button(t["predict_btn"], type="primary"):
    with st.spinner("计算中 Calculating risk..."):
        prob = float(model.predict_proba(input_df)[0, 1])
        pred_class = 1 if prob >= CUTOFF else 0

    st.subheader(t["result_header"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t["prob_label"], f"{prob:.1%}")
    with col2:
        label = t["high_risk_label"] if pred_class == 1 else t["low_risk_label"]
        st.metric(t["risk_label"], label)
    with col3:
        st.metric(t["cutoff_label"], f"{CUTOFF:.3f}")

    st.progress(int(prob * 100))
    st.caption(t["progress_caption"].format(p=f"{prob:.4f}", c=f"{CUTOFF:.4f}"))

    if pred_class == 1:
        st.warning(t["high_risk"])
    else:
        st.success(t["low_risk"])

    # ---- Per-instance SHAP explanation (graceful fallback) ----
    # Matplotlib charts use English labels only (Streamlit Cloud lacks CJK fonts).
    if explainer is not None:
        try:
            sv = explainer.shap_values(input_df)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.array(sv)[0]
            base = float(explainer.expected_value)
            order = np.argsort(np.abs(sv))[::-1]
            feat_labels = FEATURE_NAMES

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
            ax.set_xlabel("SHAP contribution (log-odds)", fontsize=9)
            ax.set_title(
                f"Per-patient attribution · base log-odds={base:.3f} → predicted probability={prob:.1%}",
                fontsize=10,
            )
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.markdown(t["shap_heading"])
            st.caption(t["shap_caption"])
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP per-patient explanation unavailable: {e}")
    else:
        st.info(t["shap_unavailable"])

    with st.expander(t["input_expander"]):
        st.json({
            t["k_ett"]: ett_24h,
            t["k_position"]: position,
            t["k_age"]: age,
            t["k_bmi"]: bmi,
            t["k_opdur"]: op_dur,
            t["k_fdp"]: fdp,
            t["k_crp"]: crp,
        })


# ----------------------------------------------------------------------------
# Sidebar footer
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(t["sidebar_footer"].format(model=t["model_label"]))


# ----------------------------------------------------------------------------
# Model explanation (global) + performance
# ----------------------------------------------------------------------------
with st.expander(t["model_exp_header"]):
    st.markdown(t["imp_heading"])
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

    st.markdown(t["perf_heading"])
    perf = pd.DataFrame({
        t["perf_metric_col"]: [
            t["m_auc"], t["m_sens"], t["m_spec"], t["m_ppv"],
            t["m_npv"], t["m_acc"], t["m_spare"],
        ],
        t["perf_value_col"]: [
            "0.764 (0.709–0.826)", "0.919 (0.841–0.960)",
            "0.424 (0.360–0.490)", "0.387", "0.929",
            "0.564", "32.7% (92/303)",
        ],
    })
    st.table(perf)
    st.caption(t["perf_caption"])

# Main page footer
st.markdown("---")
st.markdown(f"*Reference: {REFERENCE_LINE}*")
