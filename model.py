import torch
import torch.nn as nn
import torch.nn.functional as F

class PoetryModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(PoetryModel, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers = 2)
		self.linear1 = nn.Linear(self.hidden_dim, self.vocab_size)

	def forward(self, input, hidden=None):
		sentence_length, batch_size = input.size()
		if hidden == None:
			h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
			c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
		else:
			h_0, c_0 = hidden
		embeds = self.embedding(input)
		output, hidden = self.lstm(embeds, (h_0, c_0))
		output = self.linear1(output.view(sentence_length * batch_size, -1))
		return output, hidden
