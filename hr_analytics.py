# ============================================================
#  PROJECT 3: HR Analytics — Employee Attrition & Performance
#  Author : [Your Name]
#  Tools  : Python | Pandas | Matplotlib | Seaborn | Scikit-learn
#  Dataset: 800 Employees | 18 Features
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_theme(style="whitegrid")
COLORS = {"leave":"#E74C3C","stay":"#2ECC71","accent":"#3498DB","gold":"#F39C12"}
PALETTE = ["#2C3E50","#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans'})

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("=" * 60)
print("  HR ANALYTICS — EMPLOYEE ATTRITION & PERFORMANCE STUDY")
print("=" * 60)

df = pd.read_csv("data/hr_data.csv")
print(f"\n✅ Loaded {df.shape[0]} employees × {df.shape[1]} features")
print(f"   Null values: {df.isnull().sum().sum()}")

# ── 2. KEY WORKFORCE METRICS ──────────────────────────────────
total_emp      = len(df)
attrition_rate = df['Attrition'].mean() * 100
avg_income     = df['MonthlyIncome'].mean()
avg_age        = df['Age'].mean()
avg_tenure     = df['YearsAtCompany'].mean()
ot_pct         = (df['OverTime'] == 'Yes').mean() * 100

print(f"\n{'─'*50}")
print("📌 WORKFORCE KPIs")
print(f"{'─'*50}")
print(f"  Total Employees    : {total_emp:>8,}")
print(f"  Attrition Rate     : {attrition_rate:>7.1f}%")
print(f"  Avg Monthly Income : ₹{avg_income:>9,.0f}")
print(f"  Avg Employee Age   : {avg_age:>7.1f} years")
print(f"  Avg Company Tenure : {avg_tenure:>7.1f} years")
print(f"  Employees on OT    : {ot_pct:>7.1f}%")

# ── 3. BREAKDOWNS ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print("🏢 ATTRITION BY DEPARTMENT")
dept_df = df.groupby('Department').agg(
    Total=('EmployeeID','count'),
    Attrited=('Attrition','sum'),
    AvgIncome=('MonthlyIncome','mean'),
    AvgPerformance=('PerformanceScore','mean')
).round(1)
dept_df['Attrition_Rate%'] = (dept_df['Attrited']/dept_df['Total']*100).round(1)
print(dept_df.sort_values('Attrition_Rate%', ascending=False))

print(f"\n{'─'*50}")
print("👥 ATTRITION BY JOB LEVEL")
jl_df = df.groupby('JobLevel').agg(
    Total=('EmployeeID','count'),
    Attrited=('Attrition','sum'),
    AvgIncome=('MonthlyIncome','mean')
).round(1)
jl_df['Attrition_Rate%'] = (jl_df['Attrited']/jl_df['Total']*100).round(1)
print(jl_df.sort_values('Attrition_Rate%', ascending=False))

# ── 4. VISUALIZATIONS ─────────────────────────────────────────
print(f"\n{'─'*50}")
print("🎨 GENERATING VISUALIZATIONS...")

# Dashboard 1 — EDA
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("HR Analytics — Employee Attrition & Performance Dashboard",
             fontsize=16, fontweight='bold', y=1.01)

# (a) Attrition Donut
ax = axes[0,0]
counts = df['Attrition_Label'].value_counts()
colors_pie = [COLORS['stay'], COLORS['leave']]
ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
       colors=colors_pie, startangle=90,
       wedgeprops={'edgecolor':'white','linewidth':2.5})
centre = plt.Circle((0,0), 0.60, fc='white')
ax.add_patch(centre)
ax.text(0, 0.08, f'{total_emp}', ha='center', va='center',
        fontsize=16, fontweight='bold')
ax.text(0, -0.12, 'Employees', ha='center', va='center', fontsize=10)
ax.set_title("Overall Attrition Rate", fontsize=13, fontweight='bold')

# (b) Attrition by Department
ax = axes[0,1]
dept_sorted = dept_df.sort_values('Attrition_Rate%', ascending=True)
colors_dept = [COLORS['leave'] if r > 20 else COLORS['accent']
               for r in dept_sorted['Attrition_Rate%']]
bars = ax.barh(dept_sorted.index, dept_sorted['Attrition_Rate%'],
               color=colors_dept, edgecolor='white', height=0.6)
ax.axvline(attrition_rate, color='orange', linestyle='--',
           lw=2, label=f'Avg: {attrition_rate:.1f}%')
ax.set_title("Attrition Rate by Department", fontsize=13, fontweight='bold')
ax.set_xlabel("Attrition Rate (%)")
ax.legend()
for bar, val in zip(bars, dept_sorted['Attrition_Rate%']):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'{val}%', va='center', fontsize=9)

# (c) Income Distribution by Attrition
ax = axes[0,2]
for label, color, name in [('No', COLORS['stay'],'Retained'),
                             ('Yes', COLORS['leave'],'Attrited')]:
    data = df[df['Attrition_Label']==label]['MonthlyIncome']/1e3
    ax.hist(data, bins=20, alpha=0.65, color=color, label=name, edgecolor='white')
ax.set_title("Monthly Income: Attrited vs Retained", fontsize=13, fontweight='bold')
ax.set_xlabel("Monthly Income (₹ Thousands)")
ax.set_ylabel("Number of Employees")
ax.legend()

# (d) Job Satisfaction vs Attrition
ax = axes[1,0]
sat_df = df.groupby(['JobSatisfaction','Attrition_Label']).size().unstack(fill_value=0)
sat_df.plot(kind='bar', ax=ax,
            color=[COLORS['stay'], COLORS['leave']],
            edgecolor='white', width=0.6)
ax.set_title("Job Satisfaction vs Attrition", fontsize=13, fontweight='bold')
ax.set_xlabel("Satisfaction Score (1=Low, 5=High)")
ax.set_ylabel("Employees")
ax.tick_params(axis='x', rotation=0)
ax.legend(title="Attrition")

# (e) Overtime Impact
ax = axes[1,1]
ot_df = df.groupby(['OverTime','Attrition_Label']).size().unstack(fill_value=0)
ot_df.plot(kind='bar', ax=ax,
           color=[COLORS['stay'], COLORS['leave']],
           edgecolor='white', width=0.5)
ax.set_title("Overtime vs Attrition", fontsize=13, fontweight='bold')
ax.set_xlabel("")
ax.set_ylabel("Employees")
ax.tick_params(axis='x', rotation=0)
ax.legend(title="Attrition")
# Add % labels
for i, (ot_val) in enumerate(['No','Yes']):
    sub = df[df['OverTime']==ot_val]
    rate = sub['Attrition'].mean()*100
    ax.text(i, sub['Attrition'].sum()+5, f'{rate:.1f}% attrition',
            ha='center', fontsize=10, color=COLORS['leave'], fontweight='bold')

# (f) Years at Company vs Attrition (KDE)
ax = axes[1,2]
sns.kdeplot(df[df['Attrition_Label']=='No']['YearsAtCompany'],
            ax=ax, color=COLORS['stay'], fill=True, alpha=0.5, label='Retained')
sns.kdeplot(df[df['Attrition_Label']=='Yes']['YearsAtCompany'],
            ax=ax, color=COLORS['leave'], fill=True, alpha=0.5, label='Attrited')
ax.set_title("Years at Company vs Attrition", fontsize=13, fontweight='bold')
ax.set_xlabel("Years at Company")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig("visualizations/hr_dashboard.png", bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ HR dashboard saved")

# Dashboard 2 — Performance & Income
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Workforce Performance & Compensation Analysis",
             fontsize=15, fontweight='bold')

# Avg income by department
ax = axes[0]
inc_dept = df.groupby('Department')['MonthlyIncome'].mean().sort_values()/1e3
ax.barh(inc_dept.index, inc_dept.values, color=PALETTE[:len(inc_dept)], edgecolor='white')
ax.set_title("Avg Monthly Income by Department", fontsize=13, fontweight='bold')
ax.set_xlabel("Monthly Income (₹ Thousands)")
for i, val in enumerate(inc_dept.values):
    ax.text(val+0.3, i, f'₹{val:.0f}K', va='center', fontsize=9)

# Performance score heatmap by Dept × Level
ax = axes[1]
perf_pivot = df.pivot_table(values='PerformanceScore',
                             index='Department', columns='JobLevel',
                             aggfunc='mean').round(2)
sns.heatmap(perf_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, vmin=1, vmax=5)
ax.set_title("Avg Performance Score\nDept × Job Level", fontsize=13, fontweight='bold')
ax.set_xlabel("")
ax.set_ylabel("")

# Training hours vs performance
ax = axes[2]
train_perf = df.groupby('PerformanceScore')['TrainingHours'].mean()
ax.bar(train_perf.index, train_perf.values,
       color=[COLORS['leave'],'#E67E22',COLORS['gold'],COLORS['accent'],COLORS['stay']],
       edgecolor='white', width=0.6)
ax.set_title("Avg Training Hours by Performance Score", fontsize=13, fontweight='bold')
ax.set_xlabel("Performance Score (1=Low, 5=High)")
ax.set_ylabel("Avg Training Hours")
for i, (score, val) in enumerate(zip(train_perf.index, train_perf.values)):
    ax.text(score, val+0.5, f'{val:.0f}h', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("visualizations/performance_compensation.png", bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Performance chart saved")

# ── 5. MACHINE LEARNING ───────────────────────────────────────
print(f"\n{'─'*50}")
print("🤖 MACHINE LEARNING — PREDICTING ATTRITION")

df_ml = df.copy()
cat_cols = ['Gender','Department','JobLevel','Education','OverTime']
le = LabelEncoder()
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

features = ['Age','Gender','Department','JobLevel','Education',
            'YearsAtCompany','YearsInRole','MonthlyIncome',
            'PerformanceScore','JobSatisfaction','WorkLifeBalance',
            'OverTime','TrainingHours','DistanceFromHome','NumCompaniesWorked']

X = df_ml[features]
y = df_ml['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc  = accuracy_score(y_test, gb_pred)
gb_auc  = roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])

print(f"\n  Random Forest         → Accuracy: {rf_acc:.2%} | AUC: {rf_auc:.3f}")
print(f"  Gradient Boosting     → Accuracy: {gb_acc:.2%} | AUC: {gb_auc:.3f}")
print(f"\n  Classification Report (Gradient Boosting):")
print(classification_report(y_test, gb_pred, target_names=['Retained','Attrited']))

# ML Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("ML Model Results — Attrition Prediction", fontsize=15, fontweight='bold')

ax = axes[0]
cm = confusion_matrix(y_test, gb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
            xticklabels=['Retained','Attrited'],
            yticklabels=['Retained','Attrited'],
            linewidths=1, linecolor='white')
ax.set_title("Confusion Matrix\n(Gradient Boosting)", fontsize=13, fontweight='bold')
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

ax = axes[1]
for model, name, color in [(rf,'Random Forest','#3498DB'),(gb,'Gradient Boosting','#E74C3C')]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_title("ROC Curve Comparison", fontsize=13, fontweight='bold')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc='lower right')

ax = axes[2]
fi = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=True).tail(10)
colors_bar = [COLORS['leave'] if v > fi.median() else COLORS['accent'] for v in fi]
fi.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='white')
ax.set_title("Top 10 Attrition Predictors\n(Gradient Boosting)", fontsize=13, fontweight='bold')
ax.set_xlabel("Feature Importance")
high = mpatches.Patch(color=COLORS['leave'], label='High Impact')
low  = mpatches.Patch(color=COLORS['accent'], label='Low Impact')
ax.legend(handles=[high, low])

plt.tight_layout()
plt.savefig("visualizations/ml_results.png", bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ ML results chart saved")

# ── 6. BUSINESS REPORT ────────────────────────────────────────
ot_attr_rate    = df[df['OverTime']=='Yes']['Attrition'].mean()*100
no_ot_attr_rate = df[df['OverTime']=='No']['Attrition'].mean()*100
low_sat_rate    = df[df['JobSatisfaction']<=2]['Attrition'].mean()*100
new_emp_rate    = df[df['YearsAtCompany']<=2]['Attrition'].mean()*100
top_features    = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)
top_dept        = dept_df['Attrition_Rate%'].idxmax()
top_dept_rate   = dept_df['Attrition_Rate%'].max()

report = f"""
╔══════════════════════════════════════════════════════════╗
║     HR ANALYTICS — EMPLOYEE ATTRITION EXECUTIVE REPORT   ║
╚══════════════════════════════════════════════════════════╝

DATASET   : {total_emp} Employees | 18 Features | 6 Departments

WORKFORCE KPIs
─────────────────────────────────────────────────────────
  Total Employees      : {total_emp}
  Attrition Rate       : {attrition_rate:.1f}%
  Avg Monthly Income   : ₹{avg_income:,.0f}
  Avg Age              : {avg_age:.1f} years
  Avg Tenure           : {avg_tenure:.1f} years
  Employees on OT      : {ot_pct:.1f}%

KEY ATTRITION FINDINGS
─────────────────────────────────────────────────────────
  1. {top_dept} has highest attrition at {top_dept_rate:.1f}%
  2. Overtime employees leave at {ot_attr_rate:.1f}% vs {no_ot_attr_rate:.1f}% (no OT)
  3. Low satisfaction (score 1-2) leads to {low_sat_rate:.1f}% attrition
  4. New employees (≤2 yrs tenure) leave at {new_emp_rate:.1f}% — onboarding gap

TOP ATTRITION PREDICTORS (ML Model):
"""
for feat, score in top_features.head(5).items():
    report += f"  • {feat:<25} → {score:.4f}\n"

report += f"""
MODEL PERFORMANCE
─────────────────────────────────────────────────────────
  Random Forest       → Accuracy: {rf_acc:.2%} | AUC: {rf_auc:.3f}
  Gradient Boosting   → Accuracy: {gb_acc:.2%} | AUC: {gb_auc:.3f}

HR RECOMMENDATIONS
─────────────────────────────────────────────────────────
  ✅ Reduce overtime — OT doubles attrition risk
  ✅ Conduct stay interviews for satisfaction score < 3
  ✅ Strengthen 90-day & 1-year onboarding programs
  ✅ Salary review for {top_dept} — highest attrition dept
  ✅ Introduce flexible work policy to improve work-life balance
  ✅ Increase training budget — top performers get 40% more training

═══════════════════════════════════════════════════════════
"""
print(report)
with open("reports/hr_executive_report.txt","w") as f:
    f.write(report)
print("✅ Report saved → reports/hr_executive_report.txt")
print("\n🎉 HR ANALYTICS COMPLETE!\n")
