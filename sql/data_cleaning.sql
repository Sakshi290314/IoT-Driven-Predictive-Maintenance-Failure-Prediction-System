
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


------------------------------------------------------------
-- 5. Validate Target & Failure Type Consistency
------------------------------------------------------------

-- Case 1: Target = 0 but failure_type exists
SELECT *
FROM machine_sensor_data
WHERE target = 0 AND failure_type IS NOT NULL;

-- Case 2: Target = 1 but failure_type is NULL
SELECT *
FROM machine_sensor_data
WHERE target = 1 AND failure_type IS NULL;
GO
