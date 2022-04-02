import torch, json, time, sys
from tasks.CIFAR10 import CIFAR10_VIT
from tasks.ImageNet import ImageNet_ResNet50
from tasks.IMDB import IMDB_LSTM
from dvfs import CONFIG_SPACE, setDVFS
import powerLog as pl
import itertools

CONFIGS = list(itertools.product(*CONFIG_SPACE))
DEVICE = device = torch.device("cuda")
EXPERIMENTS = ['CIFAR10', 'ImageNet', 'IMDB']
RESULTS = {}

logger = pl.PowerLogger(interval=0.1, nodes=list(filter(lambda n: n[0].startswith('module/'), pl.getNodes())))


if __name__ == '__main__':

    experiment = sys.argv[1]

    print("Now profile {}".format(experiment))
    
    if experiment == 'CIFAR10':
        task_model = CIFAR10_VIT(320, 32)
    elif experiment == 'ImageNet':
        task_model = ImageNet_ResNet50(240, 8)
    else:
        task_model = IMDB_LSTM(160, 8)

    model = task_model.model.to(device)

    print('start warmup')
    for idx, (data, label) in enumerate(task_model.trainloader):
        if experiment == 'IMDB':
            data, label = label, data
        predicted_label = model(data.to(DEVICE))
        loss = task_model.criterion(predicted_label, label.to(DEVICE))
        loss.backward()
        task_model.optimizer.step()

    data, label =  next(iter(task_model.trainloader))
    if experiment == 'IMDB':
            data, label = label, data

    print('start profile')
    for config in CONFIGS:
        setDVFS(config)
        logger.start()
        t0 = time.perf_counter()
        for _ in range(10):
            predicted_label = model(data.to(DEVICE))
            loss = task_model.criterion(predicted_label, label.to(DEVICE))
            loss.backward()
            task_model.optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logger.stop()
        unitTime, unitEnergy = (t1-t0) / 10, logger.getTotalEnergy() / 10 /1000
        print(config, (unitTime, unitEnergy))
        RESULTS[','.join([str(i) for i in config])] = (unitTime, unitEnergy)
        logger.reset()

    file_name = "{}.json".format(experiment)
    with open(file_name , 'w') as fp:
        json.dump(RESULTS, fp)


    
    




