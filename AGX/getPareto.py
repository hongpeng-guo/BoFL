import json

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}


def get_Pareto(file):
    data = load_json(file)

    reverse_data = {value: key for key, value in data.items()}

    sorted_metrics = sorted(reverse_data.keys())

    pareto = [sorted_metrics[0]]

    for i in range(1, len(sorted_metrics)):
        if sorted_metrics[i][1] < pareto[-1][1]:
            pareto.append(sorted_metrics[i])

    return pareto, [reverse_data[key] for key in pareto]


if __name__ == '__main__':
    pareto, config = get_Pareto('ImageNet.json')
    for i in range(len(pareto)):
        print(pareto[i], config[i])

    data = load_json('ImageNet.json').values()
    t_data = [d[0] for d in data]
    e_data = [d[1] for d in data]
    print(max(t_data) / min(t_data))
    print(max(e_data) / min(e_data))