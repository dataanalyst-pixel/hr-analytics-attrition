# 👥 HR Analytics — Employee Attrition & Performance Study

> A comprehensive HR data analytics project studying **why employees leave**, which departments are at highest risk, and how to predict attrition using Machine Learning — built to demonstrate business-critical analytical skills.

---

## 🚀 Project Overview

Employee attrition costs companies 6–9 months of an employee's salary in recruiting and training. This project helps HR teams:
- Understand **which employees are most likely to leave**
- Identify **root causes of attrition**
- Build an **ML model to predict attrition** before it happens
- Provide **actionable HR recommendations**

---

## 🗂️ Project Structure

```
project3-hr-analytics/
│
├── data/
│   ├── hr_data.csv                   ← 800 employee dataset
│   └── generate_data.py              ← Dataset generator
│
├── sql/
│   └── hr_queries.sql                ← 10 HR SQL queries
│
├── visualizations/
│   ├── hr_dashboard.png              ← 6-panel HR dashboard
│   ├── performance_compensation.png  ← Performance & salary analysis
│   └── ml_results.png               ← Confusion matrix, ROC, features
│
├── reports/
│   └── hr_executive_report.txt       ← HR executive summary
│
├── hr_analytics.py                   ← Main Python script
├── requirements.txt
└── README.md
```

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Random Forest, Gradient Boosting) |
| Database | SQL |
| Version Control | Git, GitHub |

---

## 📁 Dataset Description

| Column | Description |
|---|---|
| `EmployeeID` | Unique identifier |
| `Age` | Employee age |
| `Gender` | Male / Female |
| `Department` | Engineering / Sales / Marketing / HR / Finance / Operations |
| `JobLevel` | Junior / Mid-Level / Senior / Manager / Director |
| `Education` | Bachelor's / Master's / PhD / Diploma |
| `YearsAtCompany` | Tenure at current company |
| `MonthlyIncome` | Salary per month |
| `PerformanceScore` | 1 (Low) to 5 (High) |
| `JobSatisfaction` | 1 (Very Low) to 5 (Very High) |
| `WorkLifeBalance` | 1 (Bad) to 4 (Excellent) |
| `OverTime` | Whether employee works overtime |
| `TrainingHours` | Annual training hours |
| `Attrition` | 0 = Stayed, 1 = Left |

---

## 📊 Key Findings

### 🔴 Attrition Risk Factors
| Factor | Attrition Rate |
|---|---|
| Low Job Satisfaction (1–2) | ~65%+ |
| Works Overtime | ~2× higher than non-OT |
| Tenure ≤ 2 Years | ~40%+ |
| Poor Work-Life Balance | ~50%+ |

### 📈 Model Performance
| Model | Accuracy | AUC Score |
|---|---|---|
| Random Forest | ~80% | ~0.85 |
| **Gradient Boosting** | **~82%** | **~0.87** |

### 💡 Top HR Recommendations
1. **Reduce overtime** — OT nearly doubles attrition risk
2. **Stay interviews** for employees with satisfaction < 3
3. **Strengthen onboarding** for first 2 years of tenure
4. **Salary review** in highest-attrition departments
5. **Flexible work policy** to improve work-life balance

---

## ▶️ How to Run

```bash
git clone https://github.com/YOUR_USERNAME/project3-hr-analytics.git
cd project3-hr-analytics
pip install -r requirements.txt
python data/generate_data.py
python hr_analytics.py
```

---

## 🎯 Skills Demonstrated

- ✅ HR Domain Knowledge
- ✅ Workforce KPI Analysis
- ✅ Multi-feature Attrition Analysis
- ✅ SQL for HR Reporting
- ✅ Two ML Models Compared (RF vs GBM)
- ✅ Performance & Compensation Analysis
- ✅ Executive HR Reporting
- ✅ Business Recommendations

---

## 👤 About Me

**[Your Name]** | Aspiring Data Analyst | Chennai, Tamil Nadu
- 📧 your.email@gmail.com
- 💼 linkedin.com/in/yourprofile
- 🐙 github.com/yourusername

⭐ **Star this repo if you found it useful!**
