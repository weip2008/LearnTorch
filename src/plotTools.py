import matplotlib.pyplot as plt

filepath = "data/SPX_TrainingData_200.csv"

row = 30 # 3 points
row = 100 # 100 points
row = 200 # 13 points
row = 10 # 245 points
row = 2 # 245 points
# row = 5710 

with open(filepath,'r') as f:
    lines = f.readlines()
values = list(map(float, lines[row].strip().split(',')))[2:]


plt.figure(figsize=(10, 6))
plt.plot(values, marker='o')
plt.title(f'Stock Sell/buy dataset (row:{row}, {len(values)} points)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

y = [1.0000, 0.7732, 0.9991, 0.8992, 0.7307, 0.7309, 0.5683, 0.7300, 0.2079,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

plt.figure(figsize=(10, 6))
plt.plot(y, marker='o')
plt.title('Padded Sequences [2,:]')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()