"""
Convient power measurement script for the Jetson TX2/Tegra X2.
"""
import os
import numpy as np

# descr, i2c-addr, channel
_nodes = [('module/main', '0041', '0'),
          ('module/cpu', '0041', '1'),
          ('module/ddr', '0041', '2'),
          ('module/gpu', '0040', '0'),
          ('module/soc', '0040', '1'),
          ('module/wifi', '0040', '2'),

          ('board/main', '0042', '0'),
          ('board/5v0-io-sys', '0042', '1'),
          ('board/3v3-sys', '0042', '2'),
          ('board/3v3-io-sleep', '0043', '0'),
          ('board/1v8-io', '0043', '1'),
          ('board/3v3-m.2', '0043', '2'),
          ]

_utils = [('cpu', '/proc/stat'),
          ('gpu', '/sys/devices/17000000.gp10b/load'),
          ('emc', '/sys/kernel/actmon_avg_activity/mc_all'),]


_valTypes = ['power', ]
_valTypesFull = ['power [mW]', ]


def getNodes():
    """Returns a list of all power measurement nodes, each a
    tuple of format (name, i2d-addr, channel)"""
    return _nodes


def getNodesByName(nameList=['module/main']):
    return [_nodes[[n[0] for n in _nodes].index(name)] for name in nameList]


def powerSensorsPresent():
    """Check whether we are on the TX2 platform/whether the sensors are present"""
    return os.path.isdir('/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/')


def getPowerMode():
    return os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]


def readValue(i2cAddr='0041', channel='0', valType='power'):
    """Reads a single value from the sensor"""
    fname = '/sys/bus/i2c/drivers/ina3221x/0-%s/iio:device%s/in_%s%s_input' % (i2cAddr, i2cAddr[-1], valType, channel)
    with open(fname, 'r') as f:
        return f.read()


def getModulePower():
    """Returns the current power consumption of the entire module in mW."""
    return float(readValue(i2cAddr='0041', channel='0', valType='power'))


def getAllValues(nodes=_nodes):
    """Returns all values (power, voltage, current) for a specific set of nodes."""
    power = [[float(readValue(i2cAddr=node[1], channel=node[2], valType=valType))
             for valType in _valTypes] for node in nodes]
    # util = [getCpuUtilization(), [getGpuUtilization()], [getEmcUtilization()]]
    return power 

def getCpuUtilization(filePath='/proc/stat'):
    """Returns the cpu utilization data extracted from '/proc/stat'"""
    with open(filePath) as cpu:
        return [float(time) for time in next(cpu).split()[1:]]


def getGpuUtilization(utilPath='/sys/devices/17000000.gp10b/load',):
    """Returns the gpu utilization data extracted for gpu"""
    with open(utilPath) as gpu_use:
        util =  float(gpu_use.read())
    return util / 10

def getEmcUtilization(utilPath='/sys/kernel/actmon_avg_activity/mc_all',
                      fullPath='/sys/kernel/debug/bpmp/debug/clk/emc/rate'):
    """Returns the gpu utilization data extracted for emc"""
    with open(utilPath) as emc_use:
        util =  float(emc_use.read())
    with open(fullPath) as emc_full:
        full =  float(emc_full.read())
    return util / full * 1e5


def printFullReport():
    """Prints a full report, i.e. (power,voltage,current) for all measurement nodes."""
    from tabulate import tabulate
    header = []
    header.append('description')
    for vt in _valTypesFull:
        header.append(vt)

    resultTable = []
    for descr, i2dAddr, channel in _nodes:
        row = []
        row.append(descr)
        for valType in _valTypes:
            row.append(readValue(i2cAddr=i2dAddr, channel=channel, valType=valType))
        resultTable.append(row)
    print(tabulate(resultTable, header))


import threading
import time


class PowerLogger:
    """This is an asynchronous power logger.
    Logging can be controlled using start(), stop().
    Special events can be marked using recordEvent().
    Results can be accessed through
    """

    def __init__(self, interval=0.01, nodes=_nodes):
        """Constructs the power logger and sets a sampling interval (default: 0.01s)
        and fixes which nodes are sampled (default: all of them)"""
        self.interval = interval
        self._startTime = -1
        self.eventLog = []
        self.dataLog = []
        self._nodes = nodes

    def start(self):
        "Starts the logging activity"""

        # define the inner function called regularly by the thread to log the data
        def threadFun():
            # start next timer
            self.start()
            # log data
            t = self._getTime() - self._startTime
            self.dataLog.append((t, getAllValues(self._nodes)))
            # ensure long enough sampling interval
            t2 = self._getTime() - self._startTime
            # assert (t2 - t < self.interval)

        # setup the timer and launch it
        self._tmr = threading.Timer(self.interval, threadFun)
        self._tmr.start()
        if self._startTime < 0:
            self._startTime = self._getTime()

    def _getTime(self):
        # return time.clock_gettime(time.CLOCK_REALTIME)
        return time.time()

    def recordEvent(self, name):
        """Records a marker a specific event (with name)"""
        t = self._getTime() - self._startTime
        self.eventLog.append((t, name))

    def stop(self):
        """Stops the logging activity"""
        self._tmr.cancel()

    def reset(self):
        """Reset the logger as newly initialized"""
        self._startTime = -1
        self.eventLog = []
        self.dataLog = []

    def getDataTrace(self, nodeName='module/main', valType='power'):
        """Return a list of sample values and time stamps for a specific measurement node and type"""
        pwrVals = [itm[1][[n[0] for n in self._nodes].index(nodeName)][_valTypes.index(valType)]
                   for itm in self.dataLog]
        timeVals = [itm[0] for itm in self.dataLog]
        return timeVals, pwrVals

    def showDataTraces(self, names=None, valType='power', showEvents=True):
        """creates a PyPlot figure showing all the measured power traces and event markers"""
        if names == None:
            names = [name for name, _, _ in self._nodes]

        # prepare data to display
        TPs = [self.getDataTrace(nodeName=name, valType=valType) for name in names]
        Ts, _ = TPs[0]
        Ps = [p for _, p in TPs]
        energies = [self.getTotalEnergy(nodeName=nodeName) for nodeName in names]
        Ps = list(map(list, zip(*Ps)))  # transpose list of lists

        # draw figure
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(Ts, Ps)
        plt.xlabel('time [s]')
        plt.ylabel(_valTypesFull[_valTypes.index(valType)])
        plt.grid(True)
        plt.legend(['%s (%.2f J)' % (name, enrgy / 1e3) for name, enrgy in zip(names, energies)])
        plt.title('power trace (NVPModel: %s)' % (os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1],))
        if showEvents:
            for t, _ in self.eventLog:
                plt.axvline(x=t, color='black')
        plt.show()
        plt.savefig('energy.jpg')

    def showMostCommonPowerValue(self, nodeName='module/main', valType='power', numBins=100):
        """computes a histogram of power values and print most frequent bin"""
        import numpy as np
        _, pwrData = np.array(self.getDataTrace(nodeName=nodeName, valType=valType))
        count, center = np.histogram(pwrData, bins=numBins)
        # import matplotlib.pyplot as plt
        # plt.bar((center[:-1]+center[1:])/2.0, count, align='center')
        maxProbVal = center[np.argmax(count)]  # 0.5*(center[np.argmax(count)] + center[np.argmax(count)+1])
        print('max frequent power bin value [mW]: %f' % (maxProbVal,))

    def getTotalEnergy(self, nodeName='module/main', valType='power'):
        """Integrate the power consumption over time."""
        timeVals, dataVals = self.getDataTrace(nodeName=nodeName, valType=valType)
        assert (len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += d * (t - tPrev)
            tPrev = t
        return wgtdSum

    def getAveragePower(self, nodeName='module/main', valType='power'):
        energy = self.getTotalEnergy(nodeName=nodeName, valType=valType)
        timeVals, _ = self.getDataTrace(nodeName=nodeName, valType=valType)
        return energy / timeVals[-1]


    def getCpuUtil(self):
        ''' 
        use /proc/stat data to calculate average cpu utilization, referece as follows:
        https://stackoverflow.com/questions/23367857/accurate-calculation-of-cpu-usage-given-in-percentage-in-linux
        '''
        user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        first = self.dataLog[0][1][-3]
        last = self.dataLog[-1][1][-3]

        PrevIdle = first[idle] 
        Idle = last[idle]

        PrevNonIdle = first[user] + first[nice] + first[system] + first[irq] + first[softirq] + first[steal] + first[iowait]
        NonIdle = last[user] + last[nice] + last[system] + last[irq] + last[softirq] + last[steal] + last[iowait]

        PrevTotal = PrevIdle + PrevNonIdle
        Total = Idle + NonIdle

        # differentiate: actual value minus the previous one
        totald = Total - PrevTotal
        idled = Idle - PrevIdle

        return 100 * (totald - idled) / totald

    def getGpuUtil(self):
        """Weighted sum of the gpu utilization over time."""
        timeVals = [item[0] for item in self.dataLog]
        dataVals = [item[1][-2][0] for item in self.dataLog]
        assert (len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += d * (t - tPrev)
            tPrev = t
        return wgtdSum / timeVals[-1]

    def getEmcUtil(self):
        """Weighted sum of the gpu utilization over time."""
        timeVals = [item[0] for item in self.dataLog]
        dataVals = [item[1][-1][0] for item in self.dataLog]
        assert (len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += d * (t - tPrev)
            tPrev = t
        return wgtdSum / timeVals[-1]

if __name__ == "__main__":
    
    # printFullReport()
    #    print(getModulePower())
    #    pl = PowerLogger(interval=0.05, nodes=getNodesByName(['module/main', 'board/main']))
    # model = two_stream_model()
    # test_data = get_static_frame_and_stacked_opt_flows()
    pl = PowerLogger(interval=0.1, nodes=list(filter(lambda n: n[0].startswith('module/'), getNodes())))

    pl.start()
    pl.recordEvent('run model!')

    # for i in range(50):
    #     result = model.predict([np.expand_dims(test_data[0], axis=0),
    # 

    time.sleep(5)

    pl.stop()
    print(pl.getCpuUtil(), pl.getGpuUtil(), pl.getEmcUtil())
    pl.showDataTraces()
