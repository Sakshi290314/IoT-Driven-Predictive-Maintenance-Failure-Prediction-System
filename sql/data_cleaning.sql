
/*
============================================================
Data Cleaning & Validation Script
Database: predictive_maintenance_db
Table: machine_sensor_data
Environment: SQL Server (SSMS)
============================================================

Objective:
This script performs data validation and cleaning before exploratory data analysis and machine learning modeling.

Steps Covered:
1. Check total records
2. Check for NULL values
3. Check for duplicate records
4. Validate sensor value ranges
5. Validate failure label consistency
============================================================
*/

USE predictive_maintenance_db;
GO

------------------------------------------------------------
-- 1. Check Total Records
------------------------------------------------------------
SELECT COUNT(*) AS total_records
FROM machine_sensor_data;
GO


------------------------------------------------------------
-- 2. Check for NULL Values
------------------------------------------------------------
SELECT *
FROM machine_sensor_data
WHERE product_id IS NULL
   OR machine_type IS NULL
   OR air_temperature_k IS NULL
   OR process_temperature_k IS NULL
   OR rotational_speed_rpm IS NULL
   OR torque_nm IS NULL
   OR tool_wear_min IS NULL
   OR target IS NULL;
GO


------------------------------------------------------------
-- 3. Check for Duplicate Records (UDI should be unique)
------------------------------------------------------------
SELECT udi, COUNT(*) AS duplicate_count
FROM machine_sensor_data
GROUP BY udi
HAVING COUNT(*) > 1;
GO


------------------------------------------------------------
-- 4. Validate Sensor Ranges
------------------------------------------------------------

-- Check invalid RPM
SELECT *
FROM machine_sensor_data
WHERE rotational_speed_rpm <= 0;

-- Check invalid Torque
SELECT *
FROM machine_sensor_data
WHERE torque_nm <= 0;

-- Check unrealistic temperature values
SELECT *
FROM machine_sensor_data
WHERE air_temperature_k < 200 OR air_temperature_k > 400
   OR process_temperature_k < 200 OR process_temperature_k > 400;
GO


/* ============================================================
   5️⃣ VALIDATE TARGET & FAILURE TYPE CONSISTENCY (CHECK FIRST)
   ============================================================ */

-- Case A: Target = 0 but failure_type exists
SELECT *
FROM machine_sensor_data
WHERE target = 0
AND failure_type IS NOT NULL;

-- Case B: Target = 1 but failure_type = 'No Failure'
SELECT *
FROM machine_sensor_data
WHERE target = 1
AND failure_type = 'No Failure';
GO


/* ============================================================
   6️⃣ STANDARDIZE FAILURE TYPE (SAFE UPDATE USING TRANSACTION)
   ============================================================ */

BEGIN TRANSACTION;

-- Case A Fix: Non-failure should not have failure_type
UPDATE machine_sensor_data
SET failure_type = NULL
WHERE target = 0
AND failure_type IS NOT NULL;

-- Case B Fix: Rename ambiguous failure
UPDATE machine_sensor_data
SET failure_type = 'Unspecified Failure'
WHERE target = 1
AND failure_type = 'No Failure';

-- Verify results after update
SELECT target, failure_type, COUNT(*) AS total_count
FROM machine_sensor_data
GROUP BY target, failure_type
ORDER BY target;

/* ============================================================
   7️⃣ CREATE CLEAN VIEW FOR EDA & MODELING
   (Avoid deleting raw data)
   ============================================================ */

CREATE OR ALTER VIEW vw_clean_machine_sensor_data AS
SELECT *
FROM machine_sensor_data
WHERE rotational_speed_rpm > 0
  AND torque_nm > 0
  AND air_temperature_k BETWEEN 200 AND 400
  AND process_temperature_k BETWEEN 200 AND 400;
GO
