import pandas as pd
import matplotlib.pyplot as plt

# Sample data for patterns
data = {
    'Datetime': ['2024-06-07 16:25:00-04:00', '2024-06-07 16:26:00-04:00', '2024-06-07 16:32:00-04:00',
                 '2024-06-07 16:33:00-04:00', '2024-06-07 16:35:00-04:00', '2024-06-07 16:36:00-04:00',
                 '2024-06-07 16:38:00-04:00', '2024-06-07 16:39:00-04:00', '2024-06-07 16:40:00-04:00',
                 '2024-06-07 16:41:00-04:00', '2024-06-07 16:43:00-04:00', '2024-06-07 16:46:00-04:00'],
    'Point': [5352.25, 5352.25, 5351.25, 5351.25, 5352.5, 5352.75, 5352.25, 5352.25, 5351.5, 5351.5, 5351.5, 5352.25],
    'Label': ['LL', 'LH', 'LH', 'LL', 'HL', 'HH', 'LH', 'LL', 'LH', 'LL', 'LH', 'HL']
}

patterns = pd.DataFrame(data)
patterns['Datetime'] = pd.to_datetime(patterns['Datetime'])
patterns.set_index('Datetime', inplace=True)
print(patterns)

# Plotting to visualize the result
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(patterns.index, patterns['Point'], label='Points')
ax.scatter(patterns.index[patterns['Label'] == 'HH'], patterns['Point'][patterns['Label'] == 'HH'], color='green', label='HH', marker='^', alpha=1)
ax.scatter(patterns.index[patterns['Label'] == 'LL'], patterns['Point'][patterns['Label'] == 'LL'], color='red', label='LL', marker='v', alpha=1)
ax.scatter(patterns.index[(patterns['Label'] == 'LH') | (patterns['Label'] == 'HL')], patterns['Point'][(patterns['Label'] == 'LH') | (patterns['Label'] == 'HL')], color='black', label='LH/HL', marker='o', alpha=1)

ax.set_title('Points with Labels')
ax.set_xlabel('Datetime')
ax.set_ylabel('Points')
ax.legend()

plt.show()
