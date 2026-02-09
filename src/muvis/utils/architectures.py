import torch.nn as nn
import torch
import math

""" Architectures for time series regression tasks.

	Conventions:
	- Input shape: (batch_size, seq_length, feat_dim)
	- Output shape: (batch_size, output_size)
	- Each architecture class should accept feat_dim as a parameter.

"""

class LSTM(nn.Module):
	def __init__(self, 
				 feat_dim, 
				 output_size = 1, 
				 hidden_size = 64, 
				 fc_units=64, 
				 num_layers=1,
				 dropout1=0.2,
				 dropout2=0.1):
		
		super(LSTM, self).__init__()

		# Dropout values
		self.dropout_rnn = nn.Dropout(dropout1)
		self.dropout_fc = nn.Dropout(dropout2)

		# LSTM Layer
		self.lstm = nn.LSTM(
			input_size=feat_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True
		)

		# Fully Connected Layer
		self.fc = nn.Linear(hidden_size, fc_units)

		# Output Layer
		self.out = nn.Linear(fc_units, output_size)

		# Activation
		self.relu = nn.ReLU()

	def forward(self, x):
		out, _ = self.lstm(x)          # (batch, seq_len, hidden)
		out = out[:, -1, :]           # last timestep
		out = self.dropout_rnn(out)

		out = self.relu(self.fc(out))
		out = self.dropout_fc(out)

		out = self.out(out)
		return out[:,-1]

class MLP(nn.Module):
	def __init__(self, 
				 feat_dim, 
				 output_size=1, 
				 num_hidden_layers=2,
				 hidden_dim=64,
				 dropout_hidden=0.1): 
		
		super(MLP, self).__init__()
   
		# Build layers dynamically based on num_hidden_layers
		layers = []
		
		# First hidden layer
		layers.append(nn.Linear(feat_dim, hidden_dim))
		layers.append(nn.ReLU())
		layers.append(nn.Dropout(dropout_hidden))
		
		# Additional hidden layers (all same size)
		for _ in range(num_hidden_layers - 1):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout_hidden))
		
		self.hidden_layers = nn.Sequential(*layers)
		
		# Output layer
		self.output_layer = nn.Linear(hidden_dim, output_size)

	def forward(self, x):
		x = self.hidden_layers(x)
		x = self.output_layer(x)
		return x[:, -1]
	


class LearnablePositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=1024):
		super(LearnablePositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
		nn.init.uniform_(self.pe, -0.02, 0.02)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		"""

		x = x + self.pe[:, :x.size(1), :]
		return self.dropout(x)

class Transformer(nn.Module):

	def __init__(self, feat_dim, seq_len, d_model, n_heads, num_layers, est_head_dim, output_size,
				 dropout_rate=0.2, est_head_dropout = 0.2, activation = "relu"):
		super(Transformer, self).__init__()

		self.feat_dim = feat_dim
		self.seq_len = seq_len
		self.d_model = d_model
		self.n_heads = n_heads
		self.num_layers = num_layers
		self.estimation_head_dim = est_head_dim
		self.dim_feedforward = d_model*4
		self.output_size = output_size
		self.dropout_rate = dropout_rate
		self.est_head_dropout = est_head_dropout
		self.activation = activation

		if self.activation == "gelu":
			self.act = nn.GELU()
		elif self.activation == "relu":
			self.act = nn.ReLU()
		else:
			print("no valid activation selection, falling back to relu")
			self.act = nn.ReLU()

		self.dropout1 = nn.Dropout(dropout_rate)
		self.project_inp = nn.Linear(feat_dim, d_model)
		self.pos_enc = LearnablePositionalEncoding(d_model, max_len=self.seq_len + 1)
		self.reg_token = nn.Parameter(torch.randn(1, 1, d_model))


		encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, self.dim_feedforward, dropout = self.dropout_rate, activation=self.activation, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)


		self.output_net = torch.nn.Sequential(
			nn.Linear(d_model, self.estimation_head_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(self.est_head_dropout),
			nn.Linear(self.estimation_head_dim, self.estimation_head_dim // 2),
			nn.ReLU(inplace=True),
			nn.Dropout(self.est_head_dropout),
			nn.Linear(self.estimation_head_dim // 2, output_size)
		)

	def forward(self, X, get_attention_maps=False):
		"""
		Args:
			X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
		Returns:
			output: (batch_size, num_classes)
		"""
		# Input Projection and Positional Embedding
		inp = self.project_inp(X) * math.sqrt(self.d_model)

		batch_size = inp.shape[0]
		reg_token = self.reg_token.expand(batch_size, -1, -1)
		inp = torch.cat((reg_token, inp), dim=1)

		inp = self.pos_enc(inp)  # add positional encoding

		output = self.transformer_encoder(inp) 

		output = self.act(output)
		output = self.dropout1(output)

		output = output[:, 0, :]

		# Estimation Head
		output = self.output_net(output)  # (batch_size, num_classes)

		return output[:,-1]
	
class ResidualBlock1D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super(ResidualBlock1D, self).__init__()
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
		self.bn1 = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU()
		
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
		self.bn2 = nn.BatchNorm1d(out_channels)

		# Shortcut connection to match dimensions if needed
		self.shortcut = nn.Sequential()
		if in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_channels, out_channels, kernel_size=1),
				nn.BatchNorm1d(out_channels)
			)

	def forward(self, x):
		residual = self.shortcut(x)
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual  # The Skip Connection
		return self.relu(out)

class ResNet1D(nn.Module):
	def __init__(self, 
				 feat_dim, 
				 output_size=1, 
				 num_filters=64, 
				 fc_units=64,
				 num_res_blocks=2, 
				 dropout1=0.2,
				 dropout2=0.1):
		
		super(ResNet1D, self).__init__()

		self.dropout_res = nn.Dropout(dropout1)
		self.dropout_fc = nn.Dropout(dropout2)

		# Initial projection to filter space
		self.input_conv = nn.Conv1d(feat_dim, num_filters, kernel_size=1)
		
		# Residual Blocks
		self.res_blocks = nn.Sequential(
			*[ResidualBlock1D(num_filters, num_filters) for _ in range(num_res_blocks)]
		)

		self.pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Linear(num_filters, fc_units)
		self.out = nn.Linear(fc_units, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		# x: (batch, seq_len, input_size)
		x = x.transpose(1, 2) # (batch, input_size, seq_len)

		# Feature extraction
		x = self.input_conv(x)
		x = self.res_blocks(x)
		x = self.dropout_res(x)

		# Global context and regression head
		x = self.pool(x).squeeze(-1)
		x = self.relu(self.fc(x))
		x = self.dropout_fc(x)
		
		out = self.out(x)
		return out[:, -1]