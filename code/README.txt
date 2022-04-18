Software Requirements:
1) Tensorflow 2.7.0
2) Python 3.9

To run:
1) Activate the environment (Conda preferred) with Tensorflow and all related toolkits (like CUDA) installed.
2) cd into the location of the files
3) arg commands to run:
	a) To TRAIN: python main.py --m train --data_dir <relative data path>
	b) To TEST (on public testing dataset): python main.py --m test --data_dir <relative data path>
	c) To PREDICT (on private testing dataset): python main.py --m predict --data_dir <relative data path>