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
    key = load_json('ImageNet.json').keys()
    gpu_values = [k[1] for k in key] 
    t_data = [d[0] for d in data]
    # t_data = [t_data[i] * gpu_values[i] for i in range(len(t_data))]
    e_data = [d[1] for d in data]
    print(max(t_data) / min(t_data))
    print(max(e_data) / min(e_data))

    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dvfs import CPU_FREQ_TABLE, GPU_FREQ_TABLE, EMC_FREQ_TABLE

    c1, c2 = CPU_FREQ_TABLE[0], CPU_FREQ_TABLE[-1]
    m1, m2 = EMC_FREQ_TABLE[0], EMC_FREQ_TABLE[-1]
    g1, g2 = GPU_FREQ_TABLE[0], GPU_FREQ_TABLE[-1]

    imagenet_dic = load_json('ImageNet.json')
    cifar10_dic = load_json('CIFAR10.json')
    imdb_dic = load_json('IMDB.json')

    t1, e1 = [], []
    t2, e2 = [], []
    for g in GPU_FREQ_TABLE[6:]:
        k1 = (c2, g, m1)
        k2 = (c2, g, m2)
        t1.append(imagenet_dic[k1][0])
        e1.append(imagenet_dic[k1][1])
        t2.append(imagenet_dic[k2][0])
        e2.append(imagenet_dic[k2][1])

    plt.clf()
    plt.plot(t1)
    plt.plot(t2)
    # plt.plot(e1)
    # plt.plot(e2)
    plt.savefig('motivation1.jpg')


    plt.clf()

    t1, e1 = [], []
    t2, e2 = [], []
    t3, e3 = [], []
    for c in CPU_FREQ_TABLE[2:]:
        k1 = (c, g2, m2)
        t1.append(imagenet_dic[k1][0])
        e1.append(imagenet_dic[k1][1])
        t2.append(cifar10_dic[k1][0])
        e2.append(cifar10_dic[k1][1])
        t3.append(imdb_dic[k1][0])
        e3.append(imdb_dic[k1][1])

    plt.plot(t1)
    plt.plot(t2)
    plt.plot(t3)
    # plt.plot(e1)
    # plt.plot(e2)
    # plt.plot(e3)
    plt.savefig('motivation2.jpg')
