import logging    

#logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to DEBUG
    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format=' %(levelname)s => %(message)s'
)

datafile = None 
#db_file = None
IsDebug = True
#WindowLen = 5

#Trainning data lenth
# average number of working days in a month is 21.7, based on a five-day workweek
# so 45 days is total for two months working days
# 200 days is one year working days
tdLen = 50

# Series Number for output training data
SN = "100"
    
# ZigZag parameters
deviation = 0.001  # Percentage
    
symbol = "SPY"
#symbol = "MES=F"

# Define the table name as a string variable
#table_name = "AAPL_1m"
table_name = "SPY_1m"
# Define the SQLite database file
data_dir = "stockdata"
