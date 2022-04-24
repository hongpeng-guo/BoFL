import os, time

CPU_FREQ_TABLE = [  345600, 499200, 652800, 806400, 
                    960000, 1113600, 1267200, 1420800, 
                    1574400, 1728000, 1881600, 2035200  ]
                    
GPU_FREQ_TABLE = [  114750000, 216750000, 318750000, 420750000,
                    522750000, 624750000, 726750000, 828750000,
                    930750000, 1032750000, 1134750000, 1236750000,
                    1300500000  ]

EMC_FREQ_TABLE = [  408000000, 665600000, 800000000, 1331200000, 
					1600000000, 1866000000  ]

CONFIG_SPACE = [CPU_FREQ_TABLE, GPU_FREQ_TABLE, EMC_FREQ_TABLE]


def setArmOnline():
	"""Set all ARM CPUs cores online"""

	for i in [3, 4, 5]:
		fname = "/sys/devices/system/cpu/cpu{:d}/online".format(i)
		with open(fname, 'w') as f:
			f.write('1')


def setDenverOnline():
	"""Set all Denver CPUs cores online"""

	for i in [1, 2]:
		fname = "/sys/devices/system/cpu/cpu{:d}/online".format(i)
		with open(fname, 'w') as f:
			f.write('1')


def setDenverOffline():
	"""Set all Denver CPUs cores offline"""

	for i in [1, 2]:
		fname = "/sys/devices/system/cpu/cpu{:d}/online".format(i)
		with open(fname, 'w') as f:
			f.write('0')


def setArmFreq(armFreq=CPU_FREQ_TABLE[-1], armFreq_cur=0):
	"""Set all ARM CPUs frequencies based on the given param"""

	for i in [0, 3, 4, 5]:
		max_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_max_freq".format(i)
		min_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_min_freq".format(i)
		
		first, second = max_fname, min_fname
		if armFreq < armFreq_cur:
			first, second = min_fname, max_fname

		with open(first, 'w') as f:
			f.write(str(armFreq))
		with open(second, 'w') as f:
			f.write(str(armFreq))


def setDenverFreq(denverFreq=CPU_FREQ_TABLE[-1], denverFreq_cur=0):
	"""Set all Denver CPUs frequencies based on the given param"""

	for i in [1, 2]:
		max_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_max_freq".format(i)
		min_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_min_freq".format(i)

		first, second = max_fname, min_fname
		if denverFreq < denverFreq_cur:
			first, second = min_fname, max_fname

		with open(first, 'w') as f:
			f.write(str(denverFreq))
		with open(second, 'w') as f:
			f.write(str(denverFreq))


def setGpuFreq(gpuFreq=GPU_FREQ_TABLE[-1], gpuFreq_cur=0):
	"""Set the GPU frequency based on the given param"""

	max_fname = "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/max_freq"
	min_fname = "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/min_freq"

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
	
	denverOn_fname = "/sys/devices/system/cpu/cpu1/online"
	denverFreq_fname = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq"
	armFreq_fname = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
	gpuFreq_fname = "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/cur_freq"
	emcFreq_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"

	denverOn, denverFreq, armFreq, gpuFreq, emcFreq = None, None, None, None, None

	with open(denverOn_fname, 'r') as f:
		denverOn = int(f.read().strip('\n'))
	with open(denverFreq_fname, 'r') as f:
		denverFreq = int(f.read().strip('\n'))
	with open(armFreq_fname, 'r') as f:
		armFreq = int(f.read().strip('\n'))
	with open(gpuFreq_fname, 'r') as f:
		gpuFreq = int(f.read().strip('\n'))
	with open(emcFreq_fname, 'r') as f:
		emcFreq = int(f.read().strip('\n'))

	return denverOn, denverFreq, armFreq, gpuFreq, emcFreq

def setDVFS(conf):
	"""Set the system knobs, which include DVFS setting on cpu gpu
	and emc, as well as CPU hotplug based on the given parameters"""
	cpuFreq, gpuFreq, emcFreq = conf
	denverOn_cur, denverFreq_cur, armFreq_cur, gpuFreq_cur, emcFreq_cur = getcurStatus()
	
	if cpuFreq != armFreq_cur:
		setArmFreq(cpuFreq, armFreq_cur)
	if gpuFreq != gpuFreq_cur:
		setGpuFreq(gpuFreq, gpuFreq_cur)
	if emcFreq != emcFreq_cur:
		setEmcFreq(emcFreq, emcFreq_cur)
	if cpuFreq > 0:
		if denverOn_cur == 0:
			setDenverOnline()
			setDenverFreq(cpuFreq, denverFreq_cur)
		elif cpuFreq != denverFreq_cur:
			setDenverFreq(cpuFreq, denverFreq_cur)
	elif denverOn_cur == 1:
		setDenverOffline()


if __name__ == "__main__":
	t0 = time.time()
	setDVFS([499200, 1134750000, 800000000])
	t1 = time.time()
	print(t1 - t0)
	setDVFS([960000,1134750000, 800000000])
	t2 = time.time()
	print(t2 - t1)
