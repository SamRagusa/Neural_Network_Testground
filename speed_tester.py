from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import time

NUM_INPUT_NEURONS = 784
NUM_OUTPUT_NEURONS = 10
NUM_EPOCHS = 50
LEARNING_RATE = .05
BATCH_SIZES = [50,100,500,1000]
GRAPH_SHAPES = [[NUM_INPUT_NEURONS,1024,NUM_OUTPUT_NEURONS], [NUM_INPUT_NEURONS,1024,2048,NUM_OUTPUT_NEURONS], [NUM_INPUT_NEURONS,[5,32,2],[5,64,2],NUM_OUTPUT_NEURONS], [NUM_INPUT_NEURONS,[5,32,2],[5,64,2],1024,NUM_OUTPUT_NEURONS]]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()


class Neural_Network:
	#input_placeholder = None
	def __init__(self, graph_shape, the_input_placeholder=None, output_placeholder=None, graph_output=None, training_step=None):
		self.graph_shape = graph_shape
		self.variables = []
		
		if the_input_placeholder == None:
			self.input_placeholder = tf.placeholder(tf.float32, [None, NUM_INPUT_NEURONS])
		else:
			self.input_placeholder = the_input_placeholder
			
		if output_placeholder == None:
			self.output_placeholder = tf.placeholder(tf.float32, [None, NUM_OUTPUT_NEURONS])
		else:
			self.output_placeholder = output_placeholder
		
		if graph_output == None:
			self.graph_output = None
		else:
			self.graph_output = graph_output
		
		if training_step == None:
			self.training_step = None
		else:
			self.training_step = training_step
		
		
			
	def create_graph_with_softmax(self, act=tf.nn.relu, pooling_method=tf.nn.avg_pool):
		def weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial)

		def bias_variable(shape):
			initial = tf.constant(0.1, shape=shape)
			return tf.Variable(initial)

		def conv2d(dont_know_what_to_name_yet, W):
			return tf.nn.conv2d(dont_know_what_to_name_yet, W, strides=[1, 1, 1, 1], padding='SAME')  

		def pool(dont_know_what_to_name_yet, pooling_length):
			return pooling_method(dont_know_what_to_name_yet, ksize=[1, pooling_length, pooling_length, 1], strides=[1, pooling_length,pooling_length, 1], padding='SAME')

		def normal_nn_layer(layer_input,input_dim, output_dim, the_activation=act):
			weights = weight_variable([input_dim, output_dim])
			biases = bias_variable([output_dim])
			activations = the_activation(tf.matmul(layer_input, weights) + biases, 'activation')
			return activations

		def convolutional_nn_layer(layer_input, num_input_features, patch_size, num_features, pooling_length, the_activation=act):
			weights = weight_variable([patch_size, 1, num_input_features, num_features])
			biases = bias_variable([num_features])
			activations = the_activation(conv2d(layer_input, weights) + biases, 'activation')
			if pooling_length == 0:
				return activations
			else:
				pooled = pool(activations, pooling_length)
				return pooled


		#should make sure network shape works, so it starts with normal layer and ends with normal layer
		#and more of that kinda stuff

		layer_input = self.input_placeholder
		pooling_divisional_counter = 1
		for j in range(len(self.graph_shape)-1):
			if type(self.graph_shape[j]) == type(0) and type(self.graph_shape[j+1]) == type(0):
				if j == len(self.graph_shape)-2:
					layer_input = normal_nn_layer(layer_input, self.graph_shape[j], self.graph_shape[j+1], tf.nn.softmax)
				else:
					layer_input = normal_nn_layer(layer_input, self.graph_shape[j], self.graph_shape[j+1])	
			elif type(self.graph_shape[j]) == type(0) and type(self.graph_shape[j+1]) != type(0):
				layer_input = convolutional_nn_layer(tf.reshape(layer_input,[-1,28,28,1]), 1, self.graph_shape[j+1][0], self.graph_shape[j+1][1], self.graph_shape[j+1][2])
				pooling_divisional_counter = pooling_divisional_counter * (self.graph_shape[j+1][2]**2)
			elif type(self.graph_shape[j]) != type(0) and type(self.graph_shape[j+1]) == type(0):
				layer_input_flat = tf.reshape(layer_input, [-1, int(self.graph_shape[0] * self.graph_shape[j][1] / pooling_divisional_counter)])
				if j == len(self.graph_shape)-2:
					layer_input = normal_nn_layer(layer_input_flat, int(self.graph_shape[0] * self.graph_shape[j][1] / pooling_divisional_counter), self.graph_shape[j+1], tf.nn.softmax)
				else:
					layer_input = normal_nn_layer(layer_input_flat, int(self.graph_shape[0] * self.graph_shape[j][1] / pooling_divisional_counter), self.graph_shape[j+1])
			else:
				layer_input = convolutional_nn_layer(layer_input, self.graph_shape[j][1], self.graph_shape[j+1][0], self.graph_shape[j+1][1], self.graph_shape[j+1][2])
				pooling_divisional_counter = pooling_divisional_counter * (self.graph_shape[j+1][2]**2)
		self.graph_output = layer_input
		
		
	def make_cross_entropy_train_step(self, learning_rate, the_optomizer=tf.train.GradientDescentOptimizer):
		#If graph_output == None then print error and exit
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.output_placeholder * tf.log(self.graph_output), reduction_indices=[1]))
		train_step = the_optomizer(learning_rate).minimize(cross_entropy)
		self.training_step = train_step
	
	def train_network(self, get_batch, learning_rate, num_epochs, batch_size):
		#if training_step == None then say you need to create a training step first and exit
		for j in range(num_epochs):
			batch_xs, batch_ys = get_batch(batch_size)
			self.training_step.run({self.input_placeholder: batch_xs, self.output_placeholder: batch_ys})
				
	def get_accuracy(self, testing_inputs, correct_outputs):
		#if graph_output == None then say you need to create a graph first
		correct_prediction = tf.equal(tf.argmax(self.graph_output, 1), tf.argmax(self.output_placeholder, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy.eval({self.input_placeholder: testing_inputs, self.output_placeholder: correct_outputs})
			


x = tf.placeholder(tf.float32, [None, NUM_INPUT_NEURONS])
y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUT_NEURONS])


for j in range(len(BATCH_SIZES)):
	temp_time = time.time()
	
	neural_networks = [Neural_Network(shape, x, y_) for shape in GRAPH_SHAPES]

	for network in neural_networks:
		network.create_graph_with_softmax()
		tf.initialize_all_variables().run()
		network.make_cross_entropy_train_step(LEARNING_RATE)
	
	tf.initialize_all_variables().run()
 
	print("Run #", j+1, " preprocessing time:              ", time.time()-temp_time)
	
	counter = 0
	for network in neural_networks:
		counter += 1
		temp_time = time.time()
		network.train_network(mnist.train.next_batch, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZES[j])
		print("Run #", j+1, " ANN #", counter ,"  training time:         ", time.time() - temp_time)

	temp_time = time.time()

	for network in neural_networks:
		network.get_accuracy(mnist.test.images, mnist.test.labels)

	print("Run #", j+1, " total testing evaluation time:   ", time.time()-temp_time)