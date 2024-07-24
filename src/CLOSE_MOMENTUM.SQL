-- 1. Ensure Indexing:
CREATE INDEX IF NOT EXISTS idx_spx_1m_datetime ON SPX_1m (Datetime);

-- 2. Create CLOSE_MOMENTUM Table:
CREATE TABLE IF NOT EXISTS "CLOSE_MOMENTUM" (
    Datetime TEXT PRIMARY KEY,
    Velocity REAL,
    Acceleration REAL
);

-- 3. Calculate and Insert Velocities:
-- Create a temporary table to store velocities
CREATE TEMPORARY TABLE temp_velocity AS
SELECT 
    a.Datetime AS Datetime,
    a.Close - b.Close AS Velocity
FROM 
    SPX_1m a
JOIN 
    SPX_1m b
ON 
    b.Datetime = (
        SELECT Datetime
        FROM SPX_1m
        WHERE Datetime < a.Datetime
        ORDER BY Datetime DESC
        LIMIT 1
    );

-- Insert velocities into the CLOSE_MOMENTUM table
INSERT INTO CLOSE_MOMENTUM (Datetime, Velocity)
SELECT Datetime, Velocity
FROM temp_velocity;

-- Insert the initial row with NULL velocity and acceleration separately
INSERT INTO CLOSE_MOMENTUM (Datetime, Velocity, Acceleration)
SELECT Datetime, NULL, NULL
FROM SPX_1m
WHERE Datetime = (
    SELECT Datetime
    FROM SPX_1m
    ORDER BY Datetime
    LIMIT 1
);

-- 4. Batch Processing for Accelerations:
-- Use a cursor to process rows in batches
PRAGMA temp_store = MEMORY;  -- Use memory for temporary storage
PRAGMA cache_size = 10000;  -- Adjust cache size for better performance

-- Use a CTE to assign row numbers and process in batches
WITH cte AS (
    SELECT 
        Datetime,
        Velocity,
        ROW_NUMBER() OVER (ORDER BY Datetime) AS row_num
    FROM 
        CLOSE_MOMENTUM
)
UPDATE CLOSE_MOMENTUM
SET Acceleration = (
    SELECT 
        cte.Velocity - prev.Velocity
    FROM 
        cte
    JOIN 
        cte AS prev
    ON 
        cte.row_num = prev.row_num + 1
    WHERE 
        CLOSE_MOMENTUM.Datetime = cte.Datetime
);

-- Clean up temporary tables
DROP TABLE IF EXISTS temp_velocity;



