"""Library for implementing cascade (sequences) of different neural modules.

Authors
 * Peter Plantinga 2020
"""

import torch
import inspect
import logging
import operator
import functools
from speechbrain.nnet.linear import Linear

logger = logging.getLogger(__name__)


class Sequential(torch.nn.ModuleDict):
    """A sequence of modules with potentially inferring shape on construction.

    If layers are passed with names

    Arguments
    ---------
    input_shape : iterable
        A list or tuple of ints or None, representing the expected shape of an
        input tensor. None represents a variable length dimension. If no
        ``input_shape`` is passed, no shape inferenced will be performed
    *layers, **named_layers
        The inputs are treated as a list of layers to be
        applied in sequence. The output shape of each layer is used to
        infer the shape of the following layer. If a tuple is returned,
        only the shape of the first element is used to determine input
        shape of the next layer (e.g. RNN returns output, hidden).

    Example
    -------
    >>> inputs = torch.rand(10, 40, 50)
    >>> model = Sequential(input_shape=inputs.shape)
    >>> model.append(Linear, n_neurons=100)
    >>> model.append(Linear, n_neurons=200)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 40, 200])
    """

    def __init__(self, *layers, input_shape=None, **named_layers):
        super().__init__()

        # Replace None dimensions with arbitrary value
        self.input_shape = input_shape
        if input_shape and None in input_shape:
            self.input_shape = list(input_shape)
            for i, dim in enumerate(self.input_shape):

                # To reduce size of dummy tensors, use 1 for batch dim
                if i == 0 and dim is None:
                    dim = 1

                # Use 64 as nice round arbitrary value, big enough that
                # halving this dimension a few times doesn't reach 1
                self.input_shape[i] = dim or 64

        # Append non-named layers
        for layer in layers:
            self.append(layer)

        # Append named layers
        for name, layer in named_layers.items():
            self.append(layer, layer_name=name)

    def append(self, layer, *args, layer_name=None, **kwargs):
        """Add a layer to the list of layers, inferring shape if necessary.

        Arguments
        ---------
        layer : A torch.nn.Module class or object
            If the layer is a class, it should accept an argument called
            ``input_shape`` which will be inferred and passed. If the layer
            is a module object, it is added as-is.
        layer_name : str
            The name of the layer, for reference. If the name is in use,
            ``_{count}`` will be appended.
        *args, **kwargs
            These are passed to the layer if it is constructed.
        """

        # Compute layer_name
        if layer_name is None:
            layer_name = str(len(self))
        elif layer_name in self:
            index = 0
            while f"{layer_name}_{index}" in self:
                index += 1
            layer_name = f"{layer_name}_{index}"

        # Check if it needs to be constructed with input shape
        if self.input_shape:
            argspec = inspect.getfullargspec(layer)
            if "input_shape" in argspec.args + argspec.kwonlyargs:
                input_shape = self.get_output_shape()
                layer = layer(*args, input_shape=input_shape, **kwargs)

        # Finally, append the layer
        self.add_module(layer_name, layer)

    def get_output_shape(self):
        dummy_input = torch.zeros(self.input_shape)
        dummy_output = self(dummy_input)
        return dummy_output.shape

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            the input tensor to run through the network.
        """
        for layer in self.values():
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        return x


class ModuleList(torch.nn.Module):
    """This class implements a wraper to torch.nn.ModuleList with a forward() method to forward all the layers sequentially.

    For some pretained model with the SpeechBrain older implementation of Sequential class, user can use this class to load those pretrained models

    Arguments
    ---------
    *layers: torch class
        torch objects to be put in a ModuleList
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def append(self, module):
        self.layers.append(module)

    def extend(self, modules):
        self.layers.extend(modules)

    def insert(self, index, module):
        self.layers.insert(module)


def ignore_init(function):
    """Wrapper function to ignore the init_params argument"""
    return lambda x, y, init_params=False: function(x, y)


class ConnectBlocks(torch.nn.Module):
    """Connect a sequence of blocks with shortcut connections.

    Note: all shortcuts start from the output of the first block,
    since the first block may change the shape significantly.

    Arguments
    ---------
    input_shape : tuple
        The shape of the
    shortcut_type : str
        One of:
        * "residual" - first block output passed to final output
        * "dense" - input of each block is from all previous blocks
        * "skip" - output of each block is passed to final output
    shortcut_projection : bool
        Only has an effect if `shortcut_type` is passed. Whether to add a
        linear projection layer to the shortcut connection before combining
        with the output, to handle different sizes.
    shortcut_combine_fn : str or function
        Either a pre-defined function (one of "add", "sub", "mul", "div",
        "avg", "cat") or a user-defined function that takes the shortcut
        and next input, and combines them, as well as `init_params`
        in case parameters need to be initialized inside of the function.

    Example
    -------
    >>> inputs = torch.rand(10, 100, 20)
    >>> model = ConnectBlocks(
    ...     input_shape=inputs.shape, shortcut_projection=True
    ... )
    >>> model.append(Linear, n_neurons=10)
    >>> model.append(Linear, n_neurons=10, end_of_block=True)
    >>> model.append(Linear, n_neurons=10)
    >>> model.append(Linear, n_neurons=10, end_of_block=True)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 100, 10])
    """

    def __init__(
        self,
        input_shape,
        shortcut_type="residual",
        shortcut_projection=False,
        shortcut_combine_fn=torch.add,
    ):
        super().__init__()

        self.first_input_shape = input_shape
        self.block_input_shape = input_shape
        self.new_block = True
        self.blocks = torch.nn.ModuleList()
        if shortcut_type not in ["residual", "dense", "skip"]:
            raise ValueError(
                "'shortcuts' must be one of 'residual', 'dense', or 'skip'"
            )
        self.shortcut_type = shortcut_type
        self.shortcut_projection = shortcut_projection
        if shortcut_projection:
            self.projections = torch.nn.ModuleList()
        self.shortcut_combine_fn = shortcut_combine_fn

    def append(self, layer, *args, **kwargs):
        """Appends the specified module to the shortcut model.

        Arguments
        ---------
        layer : torch.nn.Module class
            This layer will get initialized with *args and **kwargs. Also,
            the argument ``input_shape`` will be passed if the layer takes it.
        *args, **kwargs
            Passed unchanged to the layer **EXCEPT** the kwarg ``end_of_block``
            which is used to indicate that the shorcut should be added in.
        """
        if self.new_block:
            self.blocks.append(Sequential(input_shape=self.block_input_shape))
            self.new_block = False

        end_of_block = False
        if "end_of_block" in kwargs:
            end_of_block = kwargs["end_of_block"]
            del kwargs["end_of_block"]

        self.blocks[-1].append(layer, *args, **kwargs)

        # When we reach the end of the block, prepare to add shortcut
        if end_of_block:

            # Use dummy input to find shape of next block
            dummy_input = torch.zeros(self.block_input_shape)
            dummy_output = self.blocks[-1](dummy_input)

            # Initialize projection if necessary
            if self.shortcut_projection:
                projection_size = functools.reduce(
                    operator.mul, dummy_output.shape[2:], 1
                )

                if self.shortcut_type == "residual":
                    shape = self.first_input_shape
                    dummy_input = torch.zeros(self.first_input_shape)
                else:
                    shape = self.block_input_shape

                self.projections.append(
                    Linear(
                        n_neurons=projection_size,
                        input_shape=shape,
                        bias=False,
                        combine_dims=True,
                    )
                )

            # Prepare for next block
            self.new_block = True
            dummy_output = self._combine(dummy_input, dummy_output, -1)
            self.block_input_shape = dummy_output.shape

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor
            The inputs to the replicated modules.
        """
        shortcut = x

        for i, block in enumerate(self.blocks):
            x = block(x)

            if self.shortcut_type == "skip":
                shortcut = self._combine(shortcut, x, i)
            if self.shortcut_type == "dense":
                x = shortcut = self._combine(shortcut, x, i)
            if self.shortcut_type == "residual":
                x = self._combine(shortcut, x, i)

        if self.shortcut_type == "skip":
            return shortcut
        else:
            return x

    def _combine(self, shortcut, x, block_index=0):
        """Handle combining shortcut with outputs."""

        # Apply projection
        if self.shortcut_projection:
            shortcut = self.projections[block_index](shortcut)
            shortcut = shortcut.reshape(x.shape)

        return self.shortcut_combine_fn(shortcut, x)
