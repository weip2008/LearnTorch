import csv
import matplotlib.pyplot as plt

# Define the file path
file_path = 'stockdata/SPY_TrainingData_200_09.csv'
start_column = 2
# file_path = 'stockdata/SPY_TestingData_200_09.csv'
#start_colomn = 1
inputs = []
target_row = 15 # specify the row you want to read

with open(file_path, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for count, row in enumerate(csvreader):
        if count == target_row:
            inputs.append(tuple(map(float, row[start_column:])))
            break  # Exit the loop after reading the desired row

x = []
y = []
v = []
a = []
const = 5
for i in range(int(len(inputs[0])/6)):
    x.append(i)
    y.append(inputs[0][i*6+2])
    v.append(inputs[0][i*6+4])
    a.append(inputs[0][i*6+5])

fig, ax1 = plt.subplots()

# Plot price on primary y-axis
price_line, = ax1.plot(x, y, 'b-', label="Price")
ax1.set_xlabel('Index')
ax1.set_ylabel('Price', color='b')

# Creating a secondary y-axis
ax2 = ax1.twinx()
velocity_line, = ax2.plot(x, v, 'r-', label="Velocity")
ax2.plot([0, len(x) - 1], [0, 0], 'k--')
ax2.set_ylabel('Velocity', color='r')

# Combine legends
lines = [price_line, velocity_line]
ax1.legend(lines, [line.get_label() for line in lines], loc=2)

plt.title("Long Point")
plt.show()
