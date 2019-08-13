import sys, os
import torch
from data import get_data
from torch import nn
from tqdm import tqdm
from torchnet import meter
import ipdb

from model import *

class Config(object):
	dataset_path = "data/"	# The path of poetry dataset
	picket_file_path = "data/training-picket.npz"	# Binary files after pre-processing can be used directly for model training
	author_limit = None	# Author limit, if not None, will only learn the author's verse
	length_limit = None	# length limit, if it is not None, only the verses of the specified length will be learned.
	class_limit = "poet.tang"	# class limit, value choose[poet.tang, poet.song]
	learning_rate = 1e-3	# The model learning rate
	weight_decay = 1e-4
	use_gpu = True	# is or not use gpu
	epoch = 20	# The model training epoch.
	batch_size = 128	# model training batch size.
	# The part after the sentence that exceeds this length is discarded,
	#and the sentence smaller than this length is padding at the specified position.
	max_length = 125
	plot_every = 20
	use_env = False	# if or not use visodm
	env = 'poetry'	# visdom env
	generatea_max_length_limit = 200	# generate poetry max length.
	debug_file_path = "debugp"
	pre_training_model_path = None	# The path of pre-training model.
	prefix_words = "细雨鱼儿出,微风燕子斜。"	# Control poetry
	start_words = "闲云潭影日悠悠"	# poetry start
	acrostic = False	# Is it a Tibetan poem?
	model_prefix = "checkpoints/"

opt = Config()

def generate_poetry_by_start(model, start_words, ix2word, word2ix, prefix_words = None):
	results = list(start_words)
	start_word_length = len(start_words)
	# set first word is <START>
	input = torch.Tensor([word2ix["<START>"]]).view(1, 1).long()
	if opt.use_gpu: input = input.cuda()
	hidden = None

	if prefix_words:
		for word in prefix_words:
			output, hidden = model(input, hidden)
			input = input.data.new([word2ix[word]]).view(1, 1)

	for index in range(opt.generatea_max_length_limit):
		output, hidden = model(input, hidden)
		if index < start_word_length:
			word = results[index]
			input = input.data.new([word2ix[word]]).view(1, 1)
		else:
			top_index = output.data[0].topk(1)[1][0].item()
			gen_word = ix2word[top_index]
			results.append(gen_word)
			input = input.data.new([top_index]).view(1, 1)
		if gen_word == "<EOP>":
			del results[-1]
			break
	return results

def generate_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
	results = []
	start_word_length = len(start_words)
	input = (torch.Tensor([word2ix["<START>"]]).view(1, 1).long())
	if opt.use_gpu: input = input.cuda()
	hidden = None

	index = 0
	pre_word = "<START>"

	if prefix_words:
		for word in prefix_words:
			output, hidden = model(input, hidden)
			input = (input.data.new([word2ix[word]])).view(1, 1)

	for i in range(opt.generatea_max_length_limit):
		output, hidden = model(input, hidden)
		top_index = output.data[0].topk(1)[1][0].item()
		word = ix2word[top_index]

		if (pre_word in {u"。", u"！", "<START>"}):
			if index == start_word_length: break
			else:
				word = start_words[index]
				index += 1
				input = (input.data.new([word2ix[word]])).view(1, 1)
		else:
			input = (input.data.new([word2ix[word]])).view(1, 1)
		results.append(word)
		pre_word = word
	return results

def generate(**kwargs):
	for para, value in kwargs.items(): setattr(opt, para, value)
	data, word2ix, ix2word = get_data(opt)
	model = PoetryModel(len(word2ix), 128, 256)
	map_location = lambda s, l: s
	state_dict = torch.load(opt.pre_training_model_path, map_location=map_location)
	model.load_state_dict(state_dict)

	if opt.use_gpu: model.cuda()

	# python2 and python3 str compatibility
	if sys.version_info.major == 3:
		if opt.start_words.isprintable():
			start_words = opt.start_words
			prefix_words = opt.prefix_words if opt.prefix_words else None
		else:
			start_words = opt.start_words.encode("ascii", "surrogateescape").decode("utf-8")
			prefix_words = opt.prefix_words.encode("ascii", "surrogateescape").decode("utf-8") if opt.prefix_words else None
	else:
		start_words = opt.start_words.decode("utf-8")
		prefix_words = opt.prefix_words.decode("utf-8") if opt.prefix_words else None

	start_words = start_words.replace(",", u"，").replace(".", u"。").replace("?", u"？")
	gen_poetry = generate_acrostic if opt.acrostic else generate_poetry_by_start

	result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
	print("".join(result))

def train(**kwargs):
	print("Step trainging: model start pre training...")
	print("input parameter list: " + str(kwargs.items()))
	for para, value in kwargs.items(): setattr(opt, para, value)

	opt.device = torch.device("cuda") if opt.use_gpu else t.device("cpu")
	device = opt.device
	print("The model use device: " + str(device))

	# part of dataset
	print("now start get dataset, get data, word2ix, ix2word...")
	train_data, word2ix, ix2word = get_data(opt)
	print("get dataset finish!")
	print("dataset content:\n{}".format(train_data))
	print("word2ix size: {}".format(len(word2ix)))
	print("ix2word size: {}".format(len(ix2word)))
	# numpy array to tensor
	train_data = torch.from_numpy(train_data)
	dataloader = torch.utils.data.DataLoader(train_data, batch_size = opt.batch_size, shuffle = True, num_workers = 1)
	print("get pytorch tensor dataset finish!")

	# define model
	model = PoetryModel(len(word2ix), 128, 256)
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
	criterion = nn.CrossEntropyLoss()
	if opt.pre_training_model_path:
		model.load_state_dict(torch.load(opt.pre_training_model_path))
	model.to(device)

	loss_meter = meter.AverageValueMeter()
	if not os.path.exists(opt.model_prefix): os.mkdir(opt.model_prefix)
	for epoch in range(opt.epoch):
		print("eopch: {}/{} training...".format(epoch+1, opt.epoch))
		for batch_index, batch_data in tqdm(enumerate(dataloader), desc="train process"):
			batch_data = batch_data.long().transpose(1, 0).contiguous()
			#print("batch index: {}  batch_data size: {}".format(batch_index, batch_data.size()))
			batch_data = batch_data.to(device)
			optimizer.zero_grad()
			input_data, target_data = batch_data[:-1, :], batch_data[1:, :]
			output_data, _ = model(input_data)
			loss = criterion(output_data, target_data.view(-1))
			loss.backward()
			optimizer.step()

			loss_meter.add(loss.item())
		print("save model: {}{}_{}.pth".format(opt.model_prefix, opt.class_limit, epoch+1))
		torch.save(model.state_dict(), "{}{}_{}.pth".format(opt.model_prefix, opt.class_limit, epoch+1))

if __name__ == "__main__":
	import fire
	fire.Fire()
