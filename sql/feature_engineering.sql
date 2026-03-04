/* ============================================================
   Predictive Maintenance - Feature Engineering
   Database: predictive_maintenance_db
   Tool Used: SQL Server Management Studio (SSMS)

   Objective:
   Create engineered features to enhance predictive modeling. Raw sensor values are transformed into domain-driven indicators such as mechanical stress, temperature differential, and risk flags.

   Note:
   Raw table is not modified.
   A new analytical view is created for modeling purposes.
   ============================================================ */

USE predictive_maintenance_db;
GO


/* ============================================================
   Create Feature-Engineered Analytical View
   ============================================================ */

CREATE OR ALTER VIEW vw_feature_engineered_data AS
SELECT
    udi,
    product_id,
    machine_type,
    air_temperature_k,
    process_temperature_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
    target,

    -- Temperature difference (Overheating indicator)
    process_temperature_k - air_temperature_k 
        AS temperature_diff,

    -- Mechanical stress indicator
    rotational_speed_rpm * torque_nm 
        AS mechanical_stress,

    -- Tool wear risk category
    CASE 
        WHEN tool_wear_min < 50 THEN 'Low'
        WHEN tool_wear_min BETWEEN 50 AND 150 THEN 'Medium'
        ELSE 'High'
    END AS wear_category,

    -- High stress flag (Binary alert feature)
    CASE
        WHEN rotational_speed_rpm > 1800 
         AND torque_nm > 60 THEN 1
        ELSE 0
    END AS high_stress_flag

FROM vw_clean_machine_sensor_data;
GO
