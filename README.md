# Post-Craniotomy DVT Risk Predictor — XGBoost (7-variable)

基于多中心数据开发的 **XGBoost** 模型，用于预测颅脑肿瘤开颅术后早期下肢深静脉血栓（DVT）风险，并给出是否可安全免除超声筛查的临床建议。

🔗 **Live App:** *(部署后填写 Streamlit Cloud 链接)*

---

## 输入特征（7 项）

| 变量 | 含义 | 类型 / 编码 |
|------|------|------------|
| `ETT_24H` | 术后 24h 内气管插管 / 机械通气 | 二分类 0/1 |
| `POSITION` | 手术体位 | 0=左侧卧位, 1=右侧卧位, 2=仰卧位 |
| `AGE` | 年龄 (岁) | 连续 |
| `BMI` | 体重指数 (kg/m²) | 连续 |
| `OP_DUR` | 手术时长 (小时) | 连续 |
| `FDP` | 纤维蛋白/纤维蛋白原降解产物 (μg/mL) | 连续 |
| `CRP` | C 反应蛋白 (mg/L) | 连续 |

特征**顺序**必须与模型训练一致：`[ETT_24H, POSITION, AGE, BMI, OP_DUR, FDP, CRP]`。
所有连续变量均使用**原始值**，无 log 变换、无标准化。

---

## 决策规则

```
probability = model.predict_proba(X)[0, 1]
high_risk   = probability >= 0.07275406271219254
```

- **高风险 (≥ 切点)**：加强预防，优先安排下肢静脉超声筛查。
- **低风险 (< 切点)**：常规护理，可安全免除超声筛查
  （外部验证集 303 例中 92 例 / 32.7% 属此组，漏诊率极低）。

---

## 外部测试集表现（303 例，VTE 86 例）

| 指标 | 数值 |
|------|------|
| AUC | 0.764 (95% CI 0.709–0.826) |
| 灵敏度 | 0.919 (0.841–0.960) |
| 特异度 | 0.424 (0.360–0.490) |
| PPV | 0.387 |
| NPV | 0.929 |
| 准确率 | 0.564 |
| 免超声比例 | 32.7% (92/303) |

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `app.py` | Streamlit 主程序 |
| `XGBoost_model.pkl` | 训练好的 XGBoost 模型（pickle，含 `feature_names_in_`） |
| `requirements.txt` | Python 依赖 |
| `README.md` | 本文件 |

---

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 将本文件夹推送到 GitHub 仓库；
2. 打开 [Streamlit Cloud](https://streamlit.io/cloud)；
3. **New app** → 选择仓库 → 主文件设为 `app.py` → **Deploy**。

---

*© 2026 北京天坛医院 & 重庆大学附属肿瘤医院*
*Reference: Wang H, et al. Early Prediction of Postoperative DVT After Brain Tumor Craniotomy Using an XGBoost Model: Development and Multi-Center External Validation. (Under review).*
