import json
import torch
import torch.fx
import torch.fx.experimental
import typing
import inspect
import builtins
import operator
from typing import Dict, List, Any
from google.protobuf.struct_pb2 import NullValue
from . import torch_titan_pb2 as pb

def from_python_value(value: Any) -> pb.NodeValue:
    """Convert a Python value to a NodeValue object for protobuf serialization"""
    
    if value is None or value is inspect.Signature.empty:
        return pb.NodeValue(null_value=NullValue.NULL_VALUE)
    elif isinstance(value, bool):
        return pb.NodeValue(bool_value=value)
    elif isinstance(value, int):
        return pb.NodeValue(int_value=value)
    elif isinstance(value, float):
        return pb.NodeValue(float_value=value)
    elif isinstance(value, str):
        return pb.NodeValue(string_value=value)
    elif isinstance(value, torch.device):
        return pb.NodeValue(device_value=str(value))
    elif isinstance(value, torch.dtype):
        return pb.NodeValue(dtype_value=str(value).split('.')[-1])
    elif isinstance(value, torch.layout):
        return pb.NodeValue(layout_value=str(value))
    elif isinstance(value, torch.memory_format):
        return pb.NodeValue(memory_format_value=str(value))
    elif isinstance(value, torch.Size) or (isinstance(value, tuple) and all(isinstance(v, int) for v in value)):
        return pb.NodeValue(shape_value=pb.TensorShape(dims=list(value)))
    elif isinstance(value, torch.fx.Node):
        return pb.NodeValue(node_ref_value=value.name)
    elif isinstance(value, (list, tuple)):
        elements = [from_python_value(item) for item in value]
        return pb.NodeValue(sequence_value=pb.SequenceValue(elements=elements))
    else:
        # For other types, try to represent as a string
        print("Got unknown type:", type(value))
        try:
            return pb.NodeValue(repr_value=repr(value))
        except Exception:
            print(f"Warning: Could not serialize value of type {type(value)}, using None as fallback")
            return pb.NodeValue(null_value=NullValue.NULL_VALUE)

def dict_to_node_values(d: Dict[str, Any]) -> List[pb.NamedNodeValue]:
    """Convert a dictionary to a list of NamedNodeValue objects for protobuf serialization"""
    node_values = []
    for key, value in d.items():
        node_value = from_python_value(value)
        named_node_value = pb.NamedNodeValue(name=key, value=node_value)
        node_values.append(named_node_value)
    return node_values


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


def serialize(gm: torch.fx.GraphModule) -> pb.GraphModuleData:
    """Asynchronously serialize a GraphModule to a JSON file using dataclasses."""
    # Create the GraphData and populate it with nodes
    graph_data = pb.GraphData()
    
    for i, node in enumerate(gm.graph.nodes):
        # Convert args to NodeValue objects
        args = [from_python_value(arg) for arg in node.args]
        kwargs = dict_to_node_values(node.kwargs)
        
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
        node_data = pb.NodeData(
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
        user_preserved_attrs = dict_to_node_values(gm.meta['_user_preserved_attributes'])
    
    graph_module_data = pb.GraphModuleData(
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