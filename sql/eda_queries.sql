/* ============================================================
   Predictive Maintenance - Exploratory Data Analysis (EDA)
   Database: predictive_maintenance_db
   Table: machine_sensor_data
   ============================================================ */

USE predictive_maintenance_db;
GO

/* ============================================================
   1️⃣ Overall Dataset Overview
   ============================================================ */

-- Total Records
SELECT COUNT(*) AS total_records
FROM machine_sensor_data;

-- Total Failures vs Non-Failures
SELECT 
    target,
    COUNT(*) AS total_count
FROM machine_sensor_data
GROUP BY target;

-- Overall Failure Rate (%)
SELECT 
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) 
    AS failure_rate_percentage
FROM machine_sensor_data;



/* ============================================================
   2️⃣ Failure Distribution
   ============================================================ */

-- Count of Each Failure Type
SELECT 
    failure_type,
    COUNT(*) AS failure_count
FROM machine_sensor_data
WHERE target = 1
GROUP BY failure_type
ORDER BY failure_count DESC;

-- Percentage Distribution of Failure Types
SELECT 
    failure_type,
    COUNT(*) AS failure_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM machine_sensor_data
WHERE target = 1
GROUP BY failure_type
ORDER BY percentage DESC;



/* ============================================================
   3️⃣ Failure by Machine Type
   ============================================================ */

-- Total Machines per Type
SELECT 
    machine_type,
    COUNT(*) AS total_machines
FROM machine_sensor_data
GROUP BY machine_type;

-- Failure Count per Machine Type
SELECT 
    machine_type,
    COUNT(*) AS failure_count
FROM machine_sensor_data
WHERE target = 1
GROUP BY machine_type;

-- Failure Rate per Machine Type
SELECT 
    machine_type,
    COUNT(*) AS total_records,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS total_failures,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
    AS failure_rate_percentage
FROM machine_sensor_data
GROUP BY machine_type;



/* ============================================================
   4️⃣ Sensor Comparison: Failure vs Non-Failure
   ============================================================ */

SELECT 
    target,
    AVG(air_temperature_k) AS avg_air_temp,
    AVG(process_temperature_k) AS avg_process_temp,
    AVG(rotational_speed_rpm) AS avg_rpm,
    AVG(torque_nm) AS avg_torque,
    AVG(tool_wear_min) AS avg_tool_wear
FROM machine_sensor_data
GROUP BY target;



/* ============================================================
   5️⃣ Temperature Analysis
   ============================================================ */

-- Average Temperature Difference
SELECT 
    AVG(process_temperature_k - air_temperature_k) AS avg_temp_difference
FROM machine_sensor_data;

-- Temperature Difference vs Failure
SELECT 
    target,
    AVG(process_temperature_k - air_temperature_k) AS avg_temp_difference
FROM machine_sensor_data
GROUP BY target;

-- Max & Min Temperatures
SELECT 
    MIN(air_temperature_k) AS min_air_temp,
    MAX(air_temperature_k) AS max_air_temp,
    MIN(process_temperature_k) AS min_process_temp,
    MAX(process_temperature_k) AS max_process_temp
FROM machine_sensor_data;



/* ============================================================
   6️⃣ Tool Wear Analysis
   ============================================================ */

-- Average Tool Wear in Failures
SELECT 
    target,
    AVG(tool_wear_min) AS avg_tool_wear
FROM machine_sensor_data
GROUP BY target;

-- Failure Rate by Wear Category
SELECT 
    CASE 
        WHEN tool_wear_min < 50 THEN 'Low Wear'
        WHEN tool_wear_min BETWEEN 50 AND 150 THEN 'Medium Wear'
        ELSE 'High Wear'
    END AS wear_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS failures,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
    AS failure_rate_percentage
FROM machine_sensor_data
GROUP BY 
    CASE 
        WHEN tool_wear_min < 50 THEN 'Low Wear'
        WHEN tool_wear_min BETWEEN 50 AND 150 THEN 'Medium Wear'
        ELSE 'High Wear'
    END
ORDER BY failure_rate_percentage DESC;



/* ============================================================
   7️⃣ RPM Analysis
   ============================================================ */

-- Average RPM in Failures
SELECT 
    target,
    AVG(rotational_speed_rpm) AS avg_rpm
FROM machine_sensor_data
GROUP BY target;

-- Failure Rate by RPM Category
SELECT 
    CASE 
        WHEN rotational_speed_rpm < 1200 THEN 'Low RPM'
        WHEN rotational_speed_rpm BETWEEN 1200 AND 1800 THEN 'Medium RPM'
        ELSE 'High RPM'
    END AS rpm_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS failures,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
    AS failure_rate_percentage
FROM machine_sensor_data
GROUP BY 
    CASE 
        WHEN rotational_speed_rpm < 1200 THEN 'Low RPM'
        WHEN rotational_speed_rpm BETWEEN 1200 AND 1800 THEN 'Medium RPM'
        ELSE 'High RPM'
    END
ORDER BY failure_rate_percentage DESC;



/* ============================================================
   8️⃣ Torque Analysis
   ============================================================ */

-- Average Torque in Failures
SELECT 
    target,
    AVG(torque_nm) AS avg_torque
FROM machine_sensor_data
GROUP BY target;

-- Failure Rate by Torque Category
SELECT 
    CASE 
        WHEN torque_nm < 40 THEN 'Low Torque'
        WHEN torque_nm BETWEEN 40 AND 60 THEN 'Medium Torque'
        ELSE 'High Torque'
    END AS torque_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS failures,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
    AS failure_rate_percentage
FROM machine_sensor_data
GROUP BY 
    CASE 
        WHEN torque_nm < 40 THEN 'Low Torque'
        WHEN torque_nm BETWEEN 40 AND 60 THEN 'Medium Torque'
        ELSE 'High Torque'
    END
ORDER BY failure_rate_percentage DESC;



/* ============================================================
   9️⃣ Combined Stress Analysis
   ============================================================ */

-- High RPM + High Torque
SELECT 
    COUNT(*) AS high_stress_cases,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS failures,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
    AS failure_rate_percentage
FROM machine_sensor_data
WHERE rotational_speed_rpm > 1800
AND torque_nm > 60;



/* ============================================================
   🔟 Machine Risk Ranking
   ============================================================ */

-- Top 10 High-Risk Product IDs
SELECT TOP 10
    product_id,
    COUNT(*) AS total_records,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS failure_count
FROM machine_sensor_data
GROUP BY product_id
ORDER BY failure_count DESC;



/* ============================================================
   1️⃣1️⃣ Outlier Analysis
   ============================================================ */

SELECT 
    MIN(rotational_speed_rpm) AS min_rpm,
    MAX(rotational_speed_rpm) AS max_rpm,
    MIN(torque_nm) AS min_torque,
    MAX(torque_nm) AS max_torque,
    MIN(tool_wear_min) AS min_tool_wear,
    MAX(tool_wear_min) AS max_tool_wear
FROM machine_sensor_data;



/* ============================================================
   1️⃣2️⃣ Strongest Differentiating Sensor
   Compare averages between failure & non-failure
   ============================================================ */

SELECT 
    'Air Temp' AS sensor,
    ABS(
        MAX(CASE WHEN target = 1 THEN air_temperature_k END) -
        MAX(CASE WHEN target = 0 THEN air_temperature_k END)
    ) AS difference
FROM machine_sensor_data;
