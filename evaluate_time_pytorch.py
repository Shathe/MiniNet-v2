import numpy as np
import random
import math
import os
import argparse
import time
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument("--width", help="width", default=1024)
parser.add_argument("--height", help="height", default=512)
parser.add_argument("--cpu_version", help="Whether to use the cpu version", default=0)
args = parser.parse_args()
cpu_version = bool(int(args.cpu_version))

class Interpolate(nn.Module):
	def __init__(self, size, mode):
		super(Interpolate, self).__init__()
		self.interp = nn.functional.interpolate
		self.size = size
		self.mode = mode

	def forward(self, x):
		x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
		return x

class SeparableConv2d(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
		super(SeparableConv2d,self).__init__()
		self.conv = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
		self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)
		self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

	def forward(self,x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		x = self.pointwise(x)
		return x

class SeparableConv2d_muldil(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
		super(SeparableConv2d_muldil,self).__init__()
		self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,1,groups=in_channels,bias=bias)
		self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,dilation,dilation,groups=in_channels,bias=bias)
		self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
		self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
		self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)

	def forward(self, inputs):
		x = self.conv1(inputs)
		x = self.bn1(x)
		x = F.relu(x)
		x2 = self.conv2(inputs)
		x2 = self.bn2(x2)
		x2 = F.relu(x2)
		x += x2
		x = self.pointwise(x)
		return x

class DownsamplerBlock(nn.Module):
	def __init__(self, ninput, noutput, **kwargs):
		super(DownsamplerBlock, self).__init__(**kwargs)
		self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=False)
		self.pool = nn.MaxPool2d(2, stride=2)
		self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

	def forward(self, input):
		output = torch.cat([self.conv(input), self.pool(input)], 1)
		output = self.bn(output)
		return F.relu(output)

class moduleERS(nn.Module):
	def __init__(self, chann, dropprob, dilated, **kwargs):
		super(moduleERS, self).__init__(**kwargs)

		self.conv3x3 = SeparableConv2d(chann, chann, kernel_size=3, stride=1, padding=dilated, dilation=dilated, bias=False)
		self.bn = nn.BatchNorm2d(chann, eps=1e-03)
		self.dropout = nn.Dropout2d(dropprob)

	def forward(self, input):
		output = self.conv3x3(input)
		output = self.bn(output)

		if (self.dropout.p != 0):
			output = self.dropout(output)

		return F.relu(output + input)  


class moduleERS_muldil(nn.Module):
	def __init__(self, chann, dropprob, dilated, **kwargs):
		super(moduleERS_muldil, self).__init__(**kwargs)

		self.conv3x3 = SeparableConv2d_muldil(chann, chann, kernel_size=3, stride=1, padding=1, dilation=dilated, bias=False)
		self.bn = nn.BatchNorm2d(chann, eps=1e-03)
		self.dropout = nn.Dropout2d(dropprob)

	def forward(self, input):
		output = self.conv3x3(input)
		output = self.bn(output)

		if (self.dropout.p != 0):
			output = self.dropout(output)

		return F.relu(output + input)  


class non_bottleneck_1d(nn.Module):
	def __init__(self, chann, dropprob, dilated1, dilated, **kwargs):
		super(non_bottleneck_1d, self).__init__(**kwargs)
		self.mod1 = moduleERS(chann, dropprob, dilated1)
		self.mod2 = moduleERS(chann, dropprob, dilated)

	def forward(self, input):
		output = self.mod1(input)
		output = self.mod2(output)
		return output



class non_bottleneck_1d_muldil(nn.Module):
	def __init__(self, chann, dropprob, dilated1, dilated, **kwargs):
		super(non_bottleneck_1d_muldil, self).__init__(**kwargs)
		self.mod1 = moduleERS_muldil(chann, dropprob, dilated1)
		self.mod2 = moduleERS_muldil(chann, dropprob, dilated)

	def forward(self, input):
		output = self.mod1(input)
		output = self.mod2(output)
		return output



class Encoder(nn.Module):
	def __init__(self, **kwargs):
		super(Encoder, self).__init__(**kwargs)
		self.initial_block = DownsamplerBlock(3, 16)

		self.layers = nn.ModuleList()

		self.layers.append(DownsamplerBlock(16, 64))

		for x in range(0, 5):  # 5 times
			self.layers.append(non_bottleneck_1d(64, 0.00, 1, 1))

		self.layers.append(DownsamplerBlock(64, 128))
		
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 2))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 4))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 8))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 16))

		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 1))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 2))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 4))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25,1, 8))

	def forward(self, input, predict=False):
		output = self.initial_block(input)

		for layer in self.layers:
			output = layer(output)

		if predict:
			output = self.output_conv(output)

		return output


class Encoder_cpu(nn.Module):
	def __init__(self, **kwargs):
		super(Encoder_cpu, self).__init__(**kwargs)
		self.initial_block = DownsamplerBlock(3, 16)

		self.layers = nn.ModuleList()

		self.layers.append(DownsamplerBlock(16, 64))

		self.layers.append(non_bottleneck_1d(64, 0.00, 1, 1))

		self.layers.append(DownsamplerBlock(64, 128))

		self.layers.append(non_bottleneck_1d_muldil(128, 0.25, 1, 2))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25, 1, 4))
		self.layers.append(non_bottleneck_1d_muldil(128, 0.25, 1, 8))

	def forward(self, input, predict=False):
		output = self.initial_block(input)

		for layer in self.layers:
			output = layer(output)

		if predict:
			output = self.output_conv(output)

		return output


class UpsamplerBlock(nn.Module):
	def __init__(self, ninput, noutput, **kwargs):
		super(UpsamplerBlock, self).__init__(**kwargs)

		self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
		self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

	def forward(self, input):
		output = self.conv(input)
		output = self.bn(output)
		return F.relu(output)



class Decoder(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(Decoder, self).__init__(**kwargs)

		self.skip_input = DownsamplerBlock(3, 16)
		self.skip_input2 = DownsamplerBlock(16, 64)

		self.upsample = UpsamplerBlock(128, 64)
		self.layers = nn.ModuleList()

		self.layers.append(non_bottleneck_1d(64, 0, 1,  1))
		self.layers.append(non_bottleneck_1d(64, 0, 1,  1))

		self.output_conv = nn.ConvTranspose2d(64, num_classes, 3, stride=2, padding=0, output_padding=0, bias=True)

	def forward(self, input, image):
		output = input
		output = self.upsample(output)

		y = self.skip_input(image)
		y = self.skip_input2(y)
		output = output + y

		for layer in self.layers:
			output = layer(output)

		output = self.output_conv(output)

		return output


class Decoder_cpu(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(Decoder_cpu, self).__init__(**kwargs)

		self.skip_input = DownsamplerBlock(3, 16)
		self.skip_input2 = DownsamplerBlock(16, 64)

		self.upsample = UpsamplerBlock(128, 64)
		self.layers = nn.ModuleList()

		self.layers.append(non_bottleneck_1d(64, 0, 1,  1))

		self.output_conv = nn.ConvTranspose2d(64, num_classes, 3, stride=2, padding=0, output_padding=0, bias=True)

	def forward(self, input, image):
		output = input
		output = self.upsample(output)

		y = self.skip_input(image)
		y = self.skip_input2(y)
		output = output + y

		for layer in self.layers:
			output = layer(output)

		output = self.output_conv(output)

		return output

class Net(nn.Module):
	def __init__(self, num_classes, **kwargs):  # use encoder to pass pretrained encoder
		super(Net, self).__init__(**kwargs)

		self.encoder = Encoder()
		self.decoder = Decoder(num_classes)

	def forward(self, input, only_encode=False, return_input_size=True):
		output = self.encoder(input)  # predict=False by default
		output = self.decoder.forward(output, input)
		if return_input_size:
			output = Interpolate(size=(input.shape[2], input.shape[3]), mode='bilinear')(output)

		return output


class Net_cpu(nn.Module):
	def __init__(self, num_classes, **kwargs):  # use encoder to pass pretrained encoder
		super(Net_cpu, self).__init__(**kwargs)

		self.encoder = Encoder_cpu()
		self.decoder = Decoder_cpu(num_classes)

	def forward(self, input, only_encode=False, return_input_size=True):
		output = self.encoder(input)  # predict=False by default
		output = self.decoder.forward(output, input)
		if return_input_size:
			output = Interpolate(size=(input.shape[2], input.shape[3]), mode='bilinear')(output)

		return output


# Hyperparameter
width = int(args.width)
height = int(args.height)

if cpu_version:
	model = Net_cpu(19)
else:
	model = Net(19)

pytorch_total_params = sum(p.numel() for p in model.parameters())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable

model = model.cuda() 
model.eval()
images = torch.randn(1,3, width, height)
images = images.cuda() 

time_train = []

inputs = Variable(images)
outputs = model(inputs)



while (True):
	inputs = Variable(images)
	with torch.no_grad():
		start_time = time.time()
		# If you want to return the resolution of the input size, set return_input_size=True
		outputs = model(inputs, return_input_size=False)

	torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
	fwt = time.time() - start_time
	time_train.append(fwt)
	print ("Forward time per img: %.3f (Mean: %.3f)" % (
	fwt / 1, sum(time_train) / len(time_train) / 1))
	print('Millions of parameters: ' + str(pytorch_total_params/1000000.))
	time.sleep(1)