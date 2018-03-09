

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
		strides=[1, 2, 2, 1], padding='SAME')

def main():
	# load the mnist data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# create placeholders for input to the computational graph
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	# initialize weights and biases to be used
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))

	# reshape x to be an image input
	x_image = tf.reshape(x, [-1,28,28,1])

	# initialize weights for first convolutional layer
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	# perform convolution operator and max pooling
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# initialize weights for fully connected layer
	W_fc1 = weight_variable([14 * 14 * 32, 1024])
	b_fc1 = bias_variable([1024])

	# perform reshaping of the pooling layer and execute fully connected layer
	h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

	# initialize weights for last layer
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	# output is y_conv
	y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

	# define loss function using cross entropy
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	# train the model using gradient descent and learning rate 0.5
	#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# begin tensorflow session, initialize variables
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	# train model for 500 steps
	for i in range(500):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		print('step : ' + str(i + 1) + '\tloss: ' + str(sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1]})))

	# evaluate model by determining the accuracy
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
	main()