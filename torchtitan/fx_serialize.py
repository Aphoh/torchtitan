import json
import torch
import torch.fx
import torch.fx.experimental
import typing
import inspect
import builtins
import operator
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple, ClassVar
from typing_extensions import Self
from pure_protobuf.annotations import Field
from pure_protobuf.message import BaseMessage
from pure_protobuf.one_of import OneOf
from typing_extensions import Annotated
from google.protobuf.struct_pb2 import NullValue


@dataclass
class TensorShape(BaseMessage):
    """Represents a sequence of dimensions (e.g., tensor shape)"""
    dims: Annotated[List[int], Field(1)] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dictionary representation"""
        return {"dims": self.dims}


@dataclass
class NodeValue(BaseMessage):
    """Represents a value in a node (arguments, kwargs, etc.)"""
    
    # Define OneOf for value types
    value_oneof: ClassVar[OneOf] = OneOf()
    which_value = value_oneof.which_one_of_getter()
    
    # All possible value fields
    null_value: Annotated[Optional[int], Field(1, one_of=value_oneof)] = None  # Using int to represent NullValue enum
    bool_value: Annotated[Optional[bool], Field(2, one_of=value_oneof)] = None
    int_value: Annotated[Optional[int], Field(3, one_of=value_oneof)] = None
    float_value: Annotated[Optional[float], Field(4, one_of=value_oneof)] = None
    string_value: Annotated[Optional[str], Field(5, one_of=value_oneof)] = None
    device_value: Annotated[Optional[str], Field(6, one_of=value_oneof)] = None
    dtype_value: Annotated[Optional[str], Field(7, one_of=value_oneof)] = None
    shape_value: Annotated[Optional[TensorShape], Field(8, one_of=value_oneof)] = None
    sequence_value: Annotated[Optional[List[Self]], Field(9, one_of=value_oneof)] = None
    node_ref_value: Annotated[Optional[str], Field(10, one_of=value_oneof)] = None
    repr_value: Annotated[Optional[str], Field(11, one_of=value_oneof)] = None
    layout_value: Annotated[Optional[str], Field(12, one_of=value_oneof)] = None
    memory_format_value: Annotated[Optional[str], Field(13, one_of=value_oneof)] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dictionary representation with just the value type and actual value"""
        which = self.which_value()
        result = {"type": which, "value": self.value}
        return result
    
    @property
    def value(self) -> Any:
        """Get the value based on the value_oneof type"""
        which = self.which_value()
        
        if which == "null_value" or self.null_value is not None:
            return None
        elif which == "bool_value":
            return self.bool_value
        elif which == "int_value":
            return self.int_value
        elif which == "float_value":
            return self.float_value
        elif which == "string_value":
            return self.string_value
        elif which == "node_ref_value":
            return self.node_ref_value
        elif which == "device_value":
            return self.device_value
        elif which == "dtype_value":
            return self.dtype_value
        elif which == "layout_value":
            return self.layout_value
        elif which == "memory_format_value":
            return self.memory_format_value
        elif which == "shape_value":
            return tuple(self.shape_value.dims) if self.shape_value else None
        elif which == "sequence_value":
            return [v.value for v in self.sequence_value] if self.sequence_value else None
        elif which == "repr_value":
            return self.repr_value
        return None
    
    @classmethod
    def from_python_value(cls, value: Any) -> 'NodeValue':
        """Convert a Python value to a NodeValue object for protobuf serialization"""
        node_value = cls()
        
        if value is None or value is inspect.Signature.empty:
            node_value.null_value = int(NullValue.NULL_VALUE)
        elif isinstance(value, bool):
            node_value.bool_value = value
        elif isinstance(value, int):
            node_value.int_value = value
        elif isinstance(value, float):
            node_value.float_value = value
        elif isinstance(value, str):
            node_value.string_value = value
        elif isinstance(value, torch.device):
            node_value.device_value = str(value)
        elif isinstance(value, torch.dtype):
            node_value.dtype_value = str(value).split('.')[-1]
        elif isinstance(value, torch.layout):
            node_value.layout_value = str(value)
        elif isinstance(value, torch.memory_format):
            node_value.memory_format_value = str(value)
        elif isinstance(value, torch.Size) or (isinstance(value, tuple) and all(isinstance(v, int) for v in value)):
            node_value.shape_value = TensorShape(dims=list(value))
        elif isinstance(value, torch.fx.Node):
            node_value.node_ref_value = value.name
        elif isinstance(value, (list, tuple)):
            elements = [NodeValue.from_python_value(item) for item in value]
            node_value.sequence_value = elements
        else:
            # For other types, try to represent as a string
            print("Got unknown type:", type(value))
            try:
                node_value.repr_value = repr(value)
            except:
                print(f"Warning: Could not serialize value of type {type(value)}, using None as fallback")
                node_value.null_value = int(NullValue.NULL_VALUE)
        
        return node_value


@dataclass
class NamedNodeValue(BaseMessage):
    """Represents a named value in a node (arguments, kwargs, etc.)"""
    name: Annotated[str, Field(1)]
    value: Annotated[NodeValue, Field(2)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dictionary representation"""
        return {
            "name": self.name,
            "value": self.value.to_dict()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> List[Self]:
        """Convert a dictionary to a NamedNodeValue object"""
        named_values = []
        for k, v in d.items():
            if not isinstance(k, str):
                print(f"Warning: Skipping non-string key: {k}")
                continue
            named_value = cls(name=k, value=NodeValue.from_python_value(v))
            named_values.append(named_value)
        return named_values


@dataclass
class NodeData(BaseMessage):
    """Represents a node in the FX graph"""
    name: Annotated[str, Field(1)]
    op: Annotated[str, Field(2)]  # "placeholder", "call_function", etc.
    target: Annotated[str, Field(3)]
    args: Annotated[List[NodeValue], Field(4)] = field(default_factory=list)
    kwargs: Annotated[List[NamedNodeValue], Field(5)] = field(default_factory=list)
    output_shape: Annotated[List[int], Field(6)] = field(default_factory=list)
    output_dtype: Annotated[Optional[str], Field(7)] = None
    output_device: Annotated[Optional[str], Field(8)] = None
    output_stride: Annotated[List[int], Field(9)] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dictionary representation"""
        return {
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "args": [arg.to_dict() for arg in self.args],
            "kwargs": [kwarg.to_dict() for kwarg in self.kwargs],
            "output_shape": self.output_shape,
            "output_dtype": self.output_dtype,
            "output_device": self.output_device,
            "output_stride": self.output_stride
        }


@dataclass
class GraphData(BaseMessage):
    """Represents an FX graph"""
    nodes: Annotated[List[NodeData], Field(1)] = field(default_factory=list)
    output_node_index: Annotated[int, Field(2)] = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dictionary representation"""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "output_node_index": self.output_node_index
        }


@dataclass
class GraphModuleData(BaseMessage):
    """Represents an FX GraphModule"""
    graph: Annotated[GraphData, Field(1)]
    user_preserved_attributes: Annotated[List[NamedNodeValue], Field(2)] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dictionary representation"""
        return {
            "graph": self.graph.to_dict(),
            "user_preserved_attributes": [attr.to_dict() for attr in self.user_preserved_attributes]
        }


# Helper function to get the qualified name string for callables
def _get_qualified_name_string(target: typing.Any) -> str:
    if isinstance(target, str):
        return str(target)
    try:
        if isinstance(target, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            return target.__name__

        if hasattr(target, '__module__') and hasattr(target, '__name__'):
            if target.__module__ == 'operator' or target.__module__ == '_operator':
                return f"operator.{target.__name__}"
            if getattr(builtins, target.__name__, None) is target:
                return target.__name__

            return f"{target.__module__}.{target.__name__}"

        return repr(target)

    except Exception:
        return repr(target)


def serialize(gm: torch.fx.GraphModule) -> GraphModuleData:
    """Asynchronously serialize a GraphModule to a JSON file using dataclasses."""
    # Create the GraphData and populate it with nodes
    graph_data = GraphData()
    
    for i, node in enumerate(gm.graph.nodes):
        # Convert args to NodeValue objects
        args = [NodeValue.from_python_value(arg) for arg in node.args]
        kwargs = NamedNodeValue.from_dict(node.kwargs)
        
        # Extract metadata if available
        output_shape = None
        output_dtype = None
        output_device = None
        output_stride = None
        meta_val = node.meta.get("val")
        if isinstance(meta_val, torch.Tensor):
            output_shape = tuple(meta_val.shape)
            output_dtype = str(meta_val.dtype)
            output_device = str(meta_val.device)
            output_stride = tuple(meta_val.stride())
        
        # Create the NodeData
        node_data = NodeData(
            name=node.name,
            op=node.op,
            target=_get_qualified_name_string(node.target),
            args=args,
            kwargs=kwargs,
            output_shape=output_shape,
            output_dtype=output_dtype,
            output_device=output_device,
            output_stride=output_stride
        )
        
        graph_data.nodes.append(node_data)
        
        if node.op == 'output':
            graph_data.output_node_index = i
    
    # Create the GraphModuleData with user preserved attributes
    user_preserved_attrs = []
    if hasattr(gm, 'meta') and '_user_preserved_attributes' in gm.meta:
        user_preserved_attrs = NamedNodeValue.from_dict(gm.meta['_user_preserved_attributes'])
    
    graph_module_data = GraphModuleData(
        graph=graph_data,
        user_preserved_attributes=user_preserved_attrs
    )
    
    return graph_module_data

def serialize_to_file(gm: torch.fx.GraphModule, output_file: str) -> None:
    """Synchronously serialize a GraphModule to a JSON file."""
    data = serialize(gm)
    # Use the custom to_dict method instead of asdict
    with open(output_file, 'w') as f:
        json.dump(data.to_dict(), f, indent=2, default=str)


# Example Usage (no change needed here based on this specific error)
if __name__ == '__main__':
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.register_buffer('const_buffer', torch.randn(5))

        def forward(self, x: torch.Tensor):
            lin_out = self.linear(x)
            relu_out = torch.relu(lin_out)
            sum_out = relu_out.sum(dim=-1)
            buffer_val = self.const_buffer
            add_out = operator.add(sum_out, buffer_val)

            topk_val, indices = torch.topk(add_out, 3)

            getitem_val = topk_val[0]
            getitem_val_1 = indices[0]

            concat_val = torch.cat([relu_out, relu_out], dim=0)

            device_const = torch.device("cuda:0")
            dtype_const = torch.float64
            empty_list = []
            empty_dict = {}

            len_example = len(empty_list)

            nested_output = (getitem_val, getitem_val_1, concat_val)
            final_output_tuple = (nested_output, device_const, dtype_const, empty_list, empty_dict, len_example)

            return final_output_tuple

    m = MyModule()
    x = torch.randn(5, 10)
    def step(params, x):
        return torch.func.functional_call(m, params, x)

    gm = torch.fx.experimental.proxy_tensor.make_fx(
        step,
        tracing_mode="fake",
        decomposition_table={},
        _allow_non_fake_inputs=True,
        record_module_stack=True,
    )(m.state_dict(), x)

    for node in gm.graph.nodes:
        meta_val = node.meta.get('val')
        if meta_val is None:
            meta_val = node.meta.get('example_value')
        if isinstance(meta_val, torch.Tensor):
            if 'val' not in node.meta:
                node.meta['val'] = meta_val
        pass

    print("Original Graph:")
    print(gm.print_readable())
    print("-" * 20)

    output_file = "my_graph_module.json"
    serialize_to_file(gm, output_file)
    print(f"Successfully serialized GraphModule to {output_file}")