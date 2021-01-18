from typing import Any, Dict, List
import torch
import torch.nn as nn

from agents.Network.serialize import deserializeLayer

# if there not anymore convolutional layers, we'll want to add a flatten layer first
def isLastConv(layer_defs: List[Any], i: int):
    t = layer_defs[i]['type']

    n = None
    if i < len(layer_defs) - 1:
        n = layer_defs[i + 1]['type']

    return t == 'conv' and n != 'conv'

# This is a pretty opinionated deserializer for taking a json description and generating
# an instance of a pytorch neural network. By being opinionated, we can reduce the number
# of options that need to be specified in the json; however, we tradeoff generalizability.
# NOTE: just be aware of some of the decisions being made in this module.
class Network(nn.Module):
    def __init__(self, inputs: int, outputs: int, params: Dict[str, Any], seed: int):
        super(Network, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

        # there's no way to seed a single random call from pytorch
        # so instead, just reset the global seed again.
        torch.manual_seed(seed)

        # we will loop over the json description layer-by-layer and append to this
        # sequential model as we go.
        self.model = nn.Sequential()

        # if no layers are specified, then we will only have a linear output layer.
        # so for linear models, just don't specify any intermediate layers (i.e. "layers": [])
        layer_defs = params.get('layers', [])
        for i, layer_def in enumerate(layer_defs):
            # takes a json description of a neural network layer
            # and returns torch parameters, activation function, and the number of outputs
            # (which will be the number of inputs for the next layer)
            weights, activation, inputs = deserializeLayer(layer_def, inputs)

            # add the (uninitialized) weights and activation to our model
            self.model.add_module(f'layer-{i}-weights', weights)
            self.model.add_module(f'layer-{i}-activation', activation)

            # initialize fully connected layers with xavier init
            # if you'd rather use the pytorch defaults, feel free to delete
            # the following code.
            if layer_def['type'] == 'fc':
                # try to see if we can figure out the right gain to use.
                # some activations won't work here (silu, dsilu, etc.) so
                # just use a default gain of 1
                try:
                    gain = nn.init.calculate_gain(layer_def['act'])
                except:
                    gain = 1

                # currently only allow xavier init.
                # TODO: make sure this opinionated default matches your setting
                nn.init.xavier_uniform_(weights.weight, gain)

            # initialize the biases separately
            if weights.bias is not None:
                nn.init.normal_(weights.bias, 0, 0.1)

            # if there are no more conv layers, go ahead and flatten for follow-up fully-connected layers
            if isLastConv(layer_defs, i):
                self.model.add_module(f'layer-{i}-flatten', nn.Flatten())


        # build the output layer and enable gradients to be passed back from the output
        # all the way to the inputs. Outputs are a list so that you can easily define
        # multi-headed networks and only have gradients passed back to feature layers
        # from some of the heads (e.g. for TDRC).
        self.features = inputs
        self.output = nn.Linear(inputs, outputs, bias=params.get('output_bias', True))
        self.output_layers = [self.output]
        self.output_grads = [True]

    # add a new head to the network. Can enable/disable the gradients from this head being passed to feature layers
    def addOutput(self, outputs: int, grad: bool = True, bias: bool = True, initial_value: float = None):
        layer = nn.Linear(self.features, outputs, bias=bias)

        if initial_value is None:
            nn.init.xavier_uniform_(layer.weight)

        else:
            nn.init.constant_(layer.weight, initial_value)

        if bias:
            nn.init.zeros_(layer.bias)

        self.output_layers.append(layer)
        self.output_grads.append(grad)

        num = len(self.output_layers)
        self.add_module(f'output-{num}', layer)

        return layer

    # take inputs and returns a list of outputs (one for each head)
    # for consistency, always returns a list even for single-headed networks.
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        outs = []
        for layer, grad in zip(self.output_layers, self.output_grads):
            if grad:
                outs.append(layer(x))
            else:
                outs.append(layer(x.detach()))

        return outs

# used for copying to target networks.
def cloneNetworkWeights(fromNet: Network, toNet: Network):
    toNet.load_state_dict(fromNet.state_dict())
