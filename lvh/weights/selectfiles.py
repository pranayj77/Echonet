from os import listdir
from os.path import isfile, join
import shutil

def split(inputs,output):
	onlyfiles = [f for f in listdir(inputs) if isfile(join(inputs, f))]
	for i,f in enumerate(onlyfiles):
		if i in range(0,50):
			shutil.copy(join(inputs, f),join(output,f))
if __name__ == '__main__':
	split('/athena/sablab/scratch/bcl2004/datasets/echonet/EchoNet-LVH/Batch1','/athena/sablab/scratch/prj4005/lvhSubset')
