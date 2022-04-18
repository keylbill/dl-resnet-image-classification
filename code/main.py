import tensorflow as tf
import logging
# import torch
import os, argparse
import numpy as np
# import tensorflow.experimental.numpy as np
from Model import MyModel
from DataLoader import load_data, load_testing_images
from Configure import model_configs

parser = argparse.ArgumentParser()
parser.add_argument("--m", help="train, test or predict")
parser.add_argument("--data_dir", help="path to data")
args = parser.parse_args()


def main(_):
	parameters = tf.compat.v1.Session()
	model = MyModel(parameters, model_configs)

	if args.m == "train":
		x_train, y_train, _, _ = load_data(args.data_dir)

		model.train(x_train, y_train, 100)

	elif args.m == "test":
		# Public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test)

	elif args.m == "predict":
		# Private testing dataset
		x_test = load_testing_images(args.data_dir)
		predictions = model.predict_prob(x_test)
		np.save("../predictions.npy", predictions)


if __name__ == "__main__":
	tf.compat.v1.disable_eager_execution()
	logging.disable(logging.WARNING)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	tf.compat.v1.app.run()
