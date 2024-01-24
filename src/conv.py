import math
from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import PairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Sigmoid, Softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

def reset(nn):
	def _reset(item):
		if hasattr(item, 'reset_parameters'):
			item.reset_parameters()

	if nn is not None:
		if hasattr(nn, 'children') and len(list(nn.children())) > 0:
			for item in nn.children():
				_reset(item)
		else:
			_reset(nn)

class Conv(MessagePassing):

	def __init__(self, 
				in_channels: Union[int, Tuple[int,int]], 
				out_channels: int, 
				root_weight: bool, 
				bias: bool, 
				dropout: float,
				edge_dim: int, 
				hdim: int, 
				**kwargs):
		kwargs.setdefault('aggr', 'add')
		super(Conv, self).__init__(node_dim=0, **kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.root_weight = root_weight
		self.bias = bias
		self.dropout = dropout
		self.edge_dim = edge_dim
		self.hdim = hdim

		if isinstance(in_channels, int):
			in_channels = (in_channels, in_channels)

		self.lin_key = Linear(in_channels[0], out_channels)
		self.lin_query = Linear(in_channels[1], out_channels)
		self.lin_value = Linear(in_channels[0], out_channels)
		self.lin_skip = Linear(in_channels[1], out_channels, bias=self.bias)
		odim = in_channels[0]
		self.mlp = Sequential(Linear(self.edge_dim, self.hdim), ReLU(inplace=True), Linear(self.hdim, odim), Sigmoid())

		self.reset_parameters()

	def reset_parameters(self):
		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		self.lin_skip.reset_parameters()
		reset(self.mlp)

	def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None):
		if isinstance(x, Tensor):
			x: PairTensor = (x, x)

		out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
		out = out.view(-1, self.out_channels)

		if self.root_weight:
			x_r = self.lin_skip(x[1])
			out += x_r

		return out

	def message(self, 
				x_i: Tensor, 
				x_j: Tensor, 
				edge_attr: Tensor,
				index: Tensor, 
				ptr: OptTensor,
				size_i: Optional[int]) -> Tensor:

		x_j = self.mlp(edge_attr).view(-1, 1) * x_j
		query = self.lin_query(x_i).view(-1, self.out_channels)
		key = self.lin_key(x_j).view(-1, self.out_channels)

		alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
		alpha = softmax(alpha, index, ptr, size_i)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		value = self.lin_value(x_j).view(-1, self.out_channels)
		out = value * alpha.view(-1, 1)

		return out

	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
