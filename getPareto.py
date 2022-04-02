import json


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}

data = load_json('ImageNet.json')
reverse_data = {value: key for key, value in data.items()}

sorted_metrics = sorted(reverse_data.keys())

pareto = [sorted_metrics[0]]

for i in range(1, len(sorted_metrics)):
    if sorted_metrics[i][1] < pareto[-1][1]:
        pareto.append(sorted_metrics[i])

print(len(pareto))
for key in pareto:
    print(key, reverse_data[key])