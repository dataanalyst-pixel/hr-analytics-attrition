-- ============================================================
--  PROJECT 3: HR Analytics — Employee Attrition SQL Queries
-- ============================================================

-- QUERY 1: Workforce Overview
SELECT
    COUNT(*)                              AS Total_Employees,
    ROUND(AVG(Age), 1)                   AS Avg_Age,
    ROUND(AVG(MonthlyIncome), 0)         AS Avg_Monthly_Income,
    ROUND(AVG(YearsAtCompany), 1)        AS Avg_Tenure_Years,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1) AS Attrition_Rate_Pct
FROM employees;

-- QUERY 2: Attrition by Department
SELECT
    Department,
    COUNT(*)                                AS Total,
    SUM(Attrition)                          AS Left_Company,
    ROUND(AVG(MonthlyIncome), 0)            AS Avg_Income,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1)  AS Attrition_Rate_Pct
FROM employees
GROUP BY Department
ORDER BY Attrition_Rate_Pct DESC;

-- QUERY 3: Impact of Overtime on Attrition
SELECT
    OverTime,
    COUNT(*)                                AS Total,
    SUM(Attrition)                          AS Attrited,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1)  AS Attrition_Rate_Pct,
    ROUND(AVG(MonthlyIncome), 0)            AS Avg_Income
FROM employees
GROUP BY OverTime
ORDER BY Attrition_Rate_Pct DESC;

-- QUERY 4: Job Satisfaction vs Attrition
SELECT
    JobSatisfaction,
    COUNT(*)                                AS Total,
    SUM(Attrition)                          AS Attrited,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1)  AS Attrition_Rate_Pct
FROM employees
GROUP BY JobSatisfaction
ORDER BY JobSatisfaction ASC;

-- QUERY 5: High-Risk Employees (Still Active — Intervene Now)
SELECT
    EmployeeID,
    Department,
    JobLevel,
    YearsAtCompany,
    MonthlyIncome,
    JobSatisfaction,
    WorkLifeBalance,
    OverTime
FROM employees
WHERE
    JobSatisfaction <= 2
    AND OverTime = 'Yes'
    AND WorkLifeBalance = 1
    AND Attrition = 0
ORDER BY JobSatisfaction ASC, YearsAtCompany DESC
LIMIT 15;

-- QUERY 6: Income vs Job Level
SELECT
    JobLevel,
    COUNT(*)                    AS Employees,
    ROUND(MIN(MonthlyIncome),0) AS Min_Income,
    ROUND(AVG(MonthlyIncome),0) AS Avg_Income,
    ROUND(MAX(MonthlyIncome),0) AS Max_Income,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1) AS Attrition_Rate_Pct
FROM employees
GROUP BY JobLevel
ORDER BY Avg_Income DESC;

-- QUERY 7: Gender Diversity & Pay Equity
SELECT
    Gender,
    COUNT(*)                    AS Total,
    ROUND(AVG(MonthlyIncome),0) AS Avg_Income,
    ROUND(AVG(PerformanceScore),2) AS Avg_Performance,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1) AS Attrition_Rate_Pct
FROM employees
GROUP BY Gender;

-- QUERY 8: Training Hours vs Performance
SELECT
    PerformanceScore,
    COUNT(*)                       AS Employees,
    ROUND(AVG(TrainingHours), 1)   AS Avg_Training_Hours,
    ROUND(AVG(MonthlyIncome), 0)   AS Avg_Income,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1) AS Attrition_Rate_Pct
FROM employees
GROUP BY PerformanceScore
ORDER BY PerformanceScore DESC;

-- QUERY 9: Early Tenure Attrition (Flight Risk)
SELECT
    CASE
        WHEN YearsAtCompany <= 1  THEN '0-1 Years'
        WHEN YearsAtCompany <= 3  THEN '1-3 Years'
        WHEN YearsAtCompany <= 7  THEN '3-7 Years'
        WHEN YearsAtCompany <= 12 THEN '7-12 Years'
        ELSE '12+ Years'
    END                                    AS Tenure_Band,
    COUNT(*)                               AS Employees,
    ROUND(SUM(Attrition)*100.0/COUNT(*),1) AS Attrition_Rate_Pct,
    ROUND(AVG(MonthlyIncome),0)            AS Avg_Income
FROM employees
GROUP BY Tenure_Band
ORDER BY Attrition_Rate_Pct DESC;

-- QUERY 10: Department × Education Attrition Matrix
SELECT
    Department,
    Education,
    COUNT(*)                                 AS Total,
    ROUND(SUM(Attrition)*100.0/COUNT(*), 1)  AS Attrition_Rate_Pct
FROM employees
GROUP BY Department, Education
ORDER BY Department, Attrition_Rate_Pct DESC;
