import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot

def thediscriminator(dims):
	winit = RandomNormal(stddev=0.02)
	image = Input(shape=dims)
	depthMap = Input(shape=dims)
	club = Concatenate()([image, depthMap])
	#Colvolutions
	ir = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=winit)(club)
	ir = LeakyReLU(alpha=0.2)(ir)
	ir = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=winit)(ir)
	ir = BatchNormalization()(ir)
	ir = LeakyReLU(alpha=0.2)(ir)
	ir = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=winit)(ir)
	ir = BatchNormalization()(ir)
	ir = LeakyReLU(alpha=0.2)(ir)
	ir = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=winit)(ir)
	ir = BatchNormalization()(ir)
	ir = LeakyReLU(alpha=0.2)(ir)
	# penultimate output layer
	ir = Conv2D(512, (4,4), padding='same', kernel_initializer=winit)(ir)
	ir = BatchNormalization()(ir)
	ir = LeakyReLU(alpha=0.2)(ir)
	ir = Conv2D(1, (4,4), padding='same', kernel_initializer=winit)(ir)
	pOp = Activation('sigmoid')(ir)
	p2pmodel = Model([image, depthMap], pOp)
	opt = Adam(lr=0.0002, beta_1=0.5)
	p2pmodel.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return p2pmodel

def encoderBlock(layer_in, n_filters, batchnorm=True):
	init = RandomNormal(stddev=0.02)
	#downsampling layer
	encblk = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		encblk = BatchNormalization()(encblk, training=True)
	encblk = LeakyReLU(alpha=0.2)(encblk)
	return encblk

def decoderBlock(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	#upsampling
	decblk = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	decblk = BatchNormalization()(decblk, training=True)
	if dropout:
		decblk = Dropout(0.5)(decblk, training=True)
	decblk = Concatenate()([decblk, skip_in])
	decblk = Activation('relu')(decblk)
	return decblk

def thegenerator(dims=(256,256,3)):
	init = RandomNormal(stddev=0.02)
	#encoding layers
	ip = Input(shape=dims)
	enc1 = encoderBlock(ip, 64, batchnorm=False)
	enc2 = encoderBlock(enc1, 128)
	enc3 = encoderBlock(enc2, 256)
	enc4 = encoderBlock(enc3, 512)
	enc5 = encoderBlock(enc4, 512)
	enc6 = encoderBlock(enc5, 512)
	enc7 = encoderBlock(enc6, 512)
	#bottleneck layer
	bottleneck = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(enc7)
	bottleneck = Activation('relu')(bottleneck)
	#decoding layers
	dec1 = decoderBlock(bottleneck, enc7, 512)
	dec2 = decoderBlock(dec1, enc6, 512)
	dec3 = decoderBlock(dec2, enc5, 512)
	dec4 = decoderBlock(dec3, enc4, 512, dropout=False)
	dec5 = decoderBlock(dec4, enc3, 256, dropout=False)
	dec6 = decoderBlock(dec5, enc2, 128, dropout=False)
	dec7 = decoderBlock(dec6, enc1, 64, dropout=False)
	#output layer
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(dec7)
	op = Activation('tanh')(g)
	model = Model(ip, op)
	return model

def p2pmodel(genMod, discMod, dims):
	for layer in discMod.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	ip = Input(shape=dims)
	generatorOut = genMod(ip)
	dis_out = discMod([ip, generatorOut])
	p2pmodel = Model(ip, [dis_out, generatorOut])
	opt = Adam(lr=0.0002, beta_1=0.5)
	p2pmodel.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return p2pmodel

def tsetInputs(filename):
	data = load(filename)
	x, y = data['arr_0'], data['arr_1']
	x = (x - 127.5) / 127.5
	y = (y - 127.5) / 127.5
	return [x, y]

def realDataSynt(dataset, count, pdims):
	df1, df2 = dataset
	ra = randint(0, df1.shape[0], count)
	x, y = df1[ra], df2[ra]
	y = ones((count, pdims, pdims, 1))
	return [x, y], y

def fakeDataSynt(genMod, fdata, pdims):
	X = genMod.predict(fdata)
	y = zeros((len(X), pdims, pdims, 1))
	return X, y

def progInfo(step, genMod, dataset, exampleCount=3):
	[xtruepos1, xtruepos2], _ = realDataSynt(dataset, exampleCount, 1)
	xfalsepos2, _ = fakeDataSynt(genMod, xtruepos1, 1)
	xtruepos1 = (xtruepos1 + 1) / 2.0
	xtruepos2 = (xtruepos2 + 1) / 2.0
	xfalsepos2 = (xfalsepos2 + 1) / 2.0
	for i in range(exampleCount):
		pyplot.subplot(3, exampleCount, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(xtruepos1[i])
	for i in range(exampleCount):
		pyplot.subplot(3, exampleCount, 1 + exampleCount + i)
		pyplot.axis('off')
		pyplot.imshow(xfalsepos2[i])
	for i in range(exampleCount):
		pyplot.subplot(3, exampleCount, 1 + exampleCount*2 + i)
		pyplot.axis('off')
		pyplot.imshow(xtruepos2[i])
	pyplot.close()
	f = 'model_%06d.h5' % (step+1)
	genMod.save(f)

def train(discMod, genMod, ganMod, dataset, epochs=100, batch=1):
	n_patch = discMod.output_shape[1]
	df1, df2 = dataset
	for i in range((int(len(df1) / batch) * epochs)):
		[xtruepos1, xtruepos2], ytrue = realDataSynt(dataset, batch, n_patch)
		xfalsepos2, yfalse = fakeDataSynt(genMod, xtruepos1, n_patch)
		discLoss1 = discMod.train_on_batch([xtruepos1, xtruepos2], ytrue)
		discLoss2 = discMod.train_on_batch([xtruepos1, xfalsepos2], yfalse)
		genLoss, dntcar1, dntcar2 = ganMod.train_on_batch(xtruepos1, [ytrue, xtruepos2])
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, discLoss1, discLoss2, genLoss))
		if (i+1) % (int(len(df1) / batch) * 10) == 0:
			progInfo(i, genMod, dataset)

df = tsetInputs('data.npz')
disc = thediscriminator(df[0].shape[1:])
gens = thegenerator(df[0].shape[1:])
ganmod = p2pmodel(gens, disc, df[0].shape[1:])
train(disc, gens, ganmod, df)