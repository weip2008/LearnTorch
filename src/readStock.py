import json

class Output:
    def __init__(self, data):
        self.sell = data[0]
        self.buy = data[1]

    def __str__(self):
        return f"sell: {self.sell}, buy: {self.buy}"

class Input:
    def __init__(self, data):
        self.index = data[0]
        self.close = data[1]
        self.slope = data[2]
        self.acceleration = data[3]
        self.weekday = data[4]
        self.time = data[5]

    def __str__(self):
        return f"Index: {self.index}, Close: {self.close}, Slope: {self.slope}, Acceleration: {self.acceleration}, Weekday: {self.weekday}, Time: {self.time}"

# Load the JSON object from file
with open('doc/stock.json', 'r') as file:
    json_obj = json.load(file)

output_data = Output(json_obj["output"])
input_data = [Input(data) for data in json_obj["input"]]

print("Output:", output_data)
print("Input:")
for data in input_data:
    print(data.index, data.close)
