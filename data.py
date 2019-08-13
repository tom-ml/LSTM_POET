import os, sys
import json
import re
import numpy as np

def _parseRawData(author_limit = None, length_limit = None, dataset_path = "data/", class_limit = "poet.tang"):
	def sentenceParse(para):
		result = re.sub(u"（.*）", "", para)
		result = re.sub(u"{.*}", "", result)
		result = re.sub(u"《.*》", "", result)
		result = re.sub(u"[\]\[]", "", result)
		final_result = ""
		for char_word in result:
			if char_word not in set("0123456789-"): final_result += char_word
		final_result = re.sub(u"。。", u"。", final_result)
		return final_result

	def handleJson(filein_path):
		final_result = []
		if not os.path.exists(filein_path):
			raise ValueError("error! not found the filein path: {}".format(filein_path))
		data = json.loads(open(filein_path, "r", encoding="utf-8").read())
		for poetry_contains in data:
			poetry_data = ""
			if author_limit is not None and poetry_contains.get("author") != author_limit:
				continue
			poetry = poetry_contains.get("paragraphs")
			flag = False
			for sentence in poetry:
				sample_sentences = re.split(u"[，！。]", sentence)
				for sample_sentence in sample_sentences:
					if length_limit is not None and len(sample_sentence) != length_limit and len(sample_sentence) != 0:
						flag = True
						break
				if flag: break
			if flag: continue
			for sentence in poetry:
				poetry_data += sentence
			poetry_data = sentenceParse(poetry_data)
			if poetry_data != "": final_result.append(poetry_data)
		return final_result

	final_data = list()
	print("loading source data file...")
	for filein_name in os.listdir(dataset_path):
		if filein_name.startswith(class_limit):
			final_data.extend(handleJson(dataset_path + filein_name))
			print("[ loading file: {} ]  OK!".format(filein_name))
		else: continue
	return final_data

def pad_sentence(sentences, max_length = None, dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0.):
	if not hasattr(sentences, "__len__"):
		raise ValueError("sentence must be iterable. {}".format(sentences))
	sentence_lengths = []
	for sentence in sentences:
		if not hasattr(sentence, '__len__'):
			raise ValueError("sentence must be a iterate".format(sentence))
		sentence_lengths.append(len(sentence))
	sample_sentence_number = len(sentences)
	if max_length is None: max_length = np.max(sentence_lengths)

	# take the sample shape from the first non empty sequence
	# checking for consistency in the main loop below.
	sample_shape = tuple()
	for sentence in sentences:
		if len(sentence) > 0:
			sample_shape = np.asarray(sentence).shape[1:]
			break

	x = (np.ones((sample_sentence_number, max_length) + sample_shape) * value).astype(dtype)

	for idx, sentence in enumerate(sentences):
		if not len(sentence): continue
		if truncating == "pre":
			new_sentence = sentence[-max_length:]
		elif truncating == "post":
			new_sentence = sentence[:max_length]
		else:
			raise ValueError('Truncating type "%s" not understood' % truncating)

		# check `trunc` has expected shape
		new_sentence = np.asarray(new_sentence, dtype=dtype)
		if new_sentence.shape[1:] != sample_shape:
			raise ValueError("new_sentence {} shape not equal with sample shape: {}".format(new_sentence.shape[1:], sample_shape))
		if padding == "post":
			x[idx, :len(new_sentence)] = new_sentence
		elif padding == "pre":
			x[idx, -len(new_sentence):] = new_sentence
		else:
			raise ValueError('Padding type "%s" not understood' % padding)

	return x
	

def get_data(opt):
	print("now start get data function...")

	print("check have or haven't picket file...")
	if os.path.exists(opt.picket_file_path):
		training_data = np.load(opt.picket_file_path, allow_pickle=True)
		training_data, word2ix, ix2word = training_data["data"], training_data["word2ix"].item(), training_data["ix2word"].item()
		return training_data, word2ix, ix2word

	training_data = _parseRawData(opt.author_limit, opt.length_limit, opt.dataset_path, opt.class_limit)
	words = {_word for _sentence in training_data for _word in _sentence}
	word2ix = {_word: _ix for _ix, _word in enumerate(words)}
	word2ix["<EOP>"] = len(word2ix)
	word2ix["<START>"] = len(word2ix)
	word2ix["</s>"] = len(word2ix)
	ix2word = {_ix: _word for _word, _ix in word2ix.items()}
	print("training_data generate finish! dataset size: {}".format(len(training_data)))
	print("word2idx dict generate finish! dict size: {}".format(len(word2ix)))
	print("idx2word dict generate finish! dict size: {}".format(len(ix2word)))

	for index in range(len(training_data)):
		training_data[index] = ["<START>"] + list(training_data[index]) + ["<EOP>"]

	train_num_data = [[word2ix[_word] for _word in sentence] for sentence in training_data]
	pad_data = pad_sentence(train_num_data, max_length = opt.max_length, padding = "pre", truncating = "post", value = len(word2ix)-1)

	# save binary file
	np.savez_compressed(opt.picket_file_path, data = pad_data, word2ix = word2ix, ix2word = ix2word)
	return pad_data, word2ix, ix2word
