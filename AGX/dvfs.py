import os, time

CPU_FREQ_TABLE = [  
					# 115200, 192000, 268800, 345600, 
					422400, 499200,
					576000, 652800, 729600, 806400, 883200, 960000, 
					1036800, 1113600, 1190400, 1267200, 1344000, 1420800,
					1497600, 1574400, 1651200, 1728000, 1804800, 1881600,
					1958400, 2035200, 2112000, 2188800, 2265600 ]
                    
GPU_FREQ_TABLE = [  114750000, 216750000, 318750000, 420750000,
                    522750000, 624750000, 675750000, 828750000,
					905250000, 1032750000, 1198500000, 1236750000, 
					1338750000, 1377000000 ]

EMC_FREQ_TABLE = [  204000000, 800000000, 1333000000, 1600000000, 
					1866000000, 2133000000  ]

CONFIG_SPACE = [CPU_FREQ_TABLE, GPU_FREQ_TABLE, EMC_FREQ_TABLE]


def setCpuOnline():
	"""Set all ARM CPUs cores online"""

	for i in [0, 1, 2, 3, 4, 5, 6, 7]:
		fname = "/sys/devices/system/cpu/cpu{:d}/online".format(i)
		with open(fname, 'w') as f:
			f.write('1')


def setCpuFreq(cpuFreq=CPU_FREQ_TABLE[-1], cpuFreq_cur=0):
	"""Set all ARM CPUs frequencies based on the given param"""

	for i in [0, 1, 2, 3, 4, 5, 6, 7]:
		max_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_max_freq".format(i)
		min_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_min_freq".format(i)
		
		first, second = max_fname, min_fname
		if cpuFreq < cpuFreq_cur:
			first, second = min_fname, max_fname

		with open(first, 'w') as f:
			f.write(str(cpuFreq))
		with open(second, 'w') as f:
			f.write(str(cpuFreq))


def setGpuFreq(gpuFreq=GPU_FREQ_TABLE[-1], gpuFreq_cur=0):
	"""Set the GPU frequency based on the given param"""

	max_fname = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq"
	min_fname = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq"

	first, second = max_fname, min_fname
	if gpuFreq < gpuFreq_cur:
		first, second = min_fname, max_fname

	with open(first, 'w') as f:
		f.write(str(gpuFreq))
	with open(second, 'w') as f:
		f.write(str(gpuFreq))


def setEmcFreq(emcFreq=EMC_FREQ_TABLE[-1], emcFreq_cur=0):
	"""Set the memory frequency based on the given param"""

	lock_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked"
	state_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/state"
	rate_fname =  "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
	cap_fname = "/sys/kernel/nvpmodel_emc_cap/emc_iso_cap"

	with open(lock_fname, 'w') as f:
		f.write('1')
	with open(state_fname, 'w') as f:
		f.write('1')

	first, second = cap_fname, rate_fname
	if emcFreq < emcFreq_cur:
		first, second = rate_fname, cap_fname

	with open(first, 'w') as f:
		f.write(str(emcFreq))
	with open(second, 'w') as f:
		f.write(str(emcFreq))

def getcurStatus():
	"""Get current system knob status, including cpu/gpu/memory freqs
	as well as the hotplug status of the Denver cores"""
	
	cpuFreq_fname = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
	gpuFreq_fname = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq"
	emcFreq_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"

	cpuFreq, gpuFreq, emcFreq = None, None, None
	with open(cpuFreq_fname, 'r') as f:
		cpuFreq = int(f.read().strip('\n'))
	with open(gpuFreq_fname, 'r') as f:
		gpuFreq = int(f.read().strip('\n'))
	with open(emcFreq_fname, 'r') as f:
		emcFreq = int(f.read().strip('\n'))

	return cpuFreq, gpuFreq, emcFreq

def setDVFS(conf):
	"""Set the system knobs, which include DVFS setting on cpu gpu
	and emc, as well as CPU hotplug based on the given parameters"""
	cpuFreq, gpuFreq, emcFreq = conf
	cpuFreq_cur, gpuFreq_cur, emcFreq_cur = getcurStatus()
	
	setCpuOnline()
	if cpuFreq != cpuFreq_cur:
		setCpuFreq(cpuFreq, cpuFreq_cur)
	if gpuFreq != gpuFreq_cur:
		setGpuFreq(gpuFreq, gpuFreq_cur)
	if emcFreq != emcFreq_cur:
		setEmcFreq(emcFreq, emcFreq_cur)


if __name__ == "__main__":
	t0 = time.time()
	setDVFS([652800, 1198500000, 800000000])
	t1 = time.time()
	print(t1 - t0)
	setDVFS([1804800,1198500000, 800000000])
	t2 = time.time()
	print(t2 - t1)
