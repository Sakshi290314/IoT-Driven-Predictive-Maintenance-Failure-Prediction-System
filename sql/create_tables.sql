/*
============================================================
Predictive Maintenance Database Setup
Environment: Microsoft SQL Server (SSMS)
Database: predictive_maintenance_db
============================================================

Project Overview:
This script initializes the SQL Server database used in the
IoT-Driven Predictive Maintenance Analytics project.

The script performs the following:

1. Creates a new database.
2. Creates a structured table to store IoT sensor data.
3. Applies constraints to maintain data integrity.
4. Bulk loads the predictive maintenance dataset from a CSV file.

Execution Environment:
- Microsoft SQL Server Management Studio (SSMS)

Before Running:
- Ensure SQL Server is running.
- Place the CSV file in an accessible folder (e.g., C:\temp\).
- Update the file path in the BULK INSERT command if needed.
============================================================
*/

-- Step 1: Create Database
CREATE DATABASE predictive_maintenance_db;
GO

-- Step 2: Switch to Database
USE predictive_maintenance_db;
GO

-- Step 3: Create Table
CREATE TABLE machine_sensor_data (
    
    udi INT PRIMARY KEY,
    
    product_id VARCHAR(20) NOT NULL,
    
    machine_type CHAR(1) CHECK (machine_type IN ('L','M','H')),
    
    air_temperature_k FLOAT NOT NULL,
    
    process_temperature_k FLOAT NOT NULL,
    
    rotational_speed_rpm INT NOT NULL CHECK (rotational_speed_rpm > 0),
    
    torque_nm FLOAT NOT NULL CHECK (torque_nm > 0),
    
    tool_wear_min INT NOT NULL CHECK (tool_wear_min >= 0),
    
    target INT NOT NULL CHECK (target IN (0,1)),
    
    failure_type VARCHAR(50)
);

GO

-- Step 4: Bulk Load Data
/*
If you receive a file access error:
- Move the CSV file to C:\temp\
- OR grant SQL Server permission to access your folder.
*/

BULK INSERT machine_sensor_data
FROM 'C:\temp\predictive_maintenance.csv'
WITH
(
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);

GO
