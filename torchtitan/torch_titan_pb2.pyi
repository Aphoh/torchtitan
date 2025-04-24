from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GraphData(_message.Message):
    __slots__ = ["nodes", "output_node_index"]
    NODES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NODE_INDEX_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeData]
    output_node_index: int
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeData, _Mapping]]] = ..., output_node_index: _Optional[int] = ...) -> None: ...

class GraphModuleData(_message.Message):
    __slots__ = ["graph", "user_preserved_attributes"]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    USER_PRESERVED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    graph: GraphData
    user_preserved_attributes: _containers.RepeatedCompositeFieldContainer[NamedNodeValue]
    def __init__(self, graph: _Optional[_Union[GraphData, _Mapping]] = ..., user_preserved_attributes: _Optional[_Iterable[_Union[NamedNodeValue, _Mapping]]] = ...) -> None: ...

class NamedNodeValue(_message.Message):
    __slots__ = ["name", "value"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: NodeValue
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[NodeValue, _Mapping]] = ...) -> None: ...

class NodeData(_message.Message):
    __slots__ = ["args", "kwargs", "name", "op", "output_device", "output_dtype", "output_shape", "output_stride", "target"]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DEVICE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DTYPE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SHAPE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_STRIDE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    args: _containers.RepeatedCompositeFieldContainer[NodeValue]
    kwargs: _containers.RepeatedCompositeFieldContainer[NamedNodeValue]
    name: str
    op: str
    output_device: str
    output_dtype: str
    output_shape: _containers.RepeatedScalarFieldContainer[int]
    output_stride: _containers.RepeatedScalarFieldContainer[int]
    target: str
    def __init__(self, name: _Optional[str] = ..., op: _Optional[str] = ..., target: _Optional[str] = ..., args: _Optional[_Iterable[_Union[NodeValue, _Mapping]]] = ..., kwargs: _Optional[_Iterable[_Union[NamedNodeValue, _Mapping]]] = ..., output_shape: _Optional[_Iterable[int]] = ..., output_dtype: _Optional[str] = ..., output_device: _Optional[str] = ..., output_stride: _Optional[_Iterable[int]] = ...) -> None: ...

class NodeValue(_message.Message):
    __slots__ = ["bool_value", "device_value", "dtype_value", "float_value", "int_value", "layout_value", "memory_format_value", "node_ref_value", "null_value", "repr_value", "sequence_value", "shape_value", "string_value"]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VALUE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_VALUE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FORMAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    NODE_REF_VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_VALUE_FIELD_NUMBER: _ClassVar[int]
    REPR_VALUE_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_VALUE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    bool_value: bool
    device_value: str
    dtype_value: str
    float_value: float
    int_value: int
    layout_value: str
    memory_format_value: str
    node_ref_value: str
    null_value: _struct_pb2.NullValue
    repr_value: str
    sequence_value: SequenceValue
    shape_value: TensorShape
    string_value: str
    def __init__(self, null_value: _Optional[_Union[_struct_pb2.NullValue, str]] = ..., bool_value: bool = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ..., string_value: _Optional[str] = ..., device_value: _Optional[str] = ..., dtype_value: _Optional[str] = ..., layout_value: _Optional[str] = ..., memory_format_value: _Optional[str] = ..., shape_value: _Optional[_Union[TensorShape, _Mapping]] = ..., sequence_value: _Optional[_Union[SequenceValue, _Mapping]] = ..., node_ref_value: _Optional[str] = ..., repr_value: _Optional[str] = ...) -> None: ...

class ParallelConfig(_message.Message):
    __slots__ = ["cp", "dp_replicate", "dp_shard", "enable_loss_parallel", "pp", "tp", "world_size"]
    CP_FIELD_NUMBER: _ClassVar[int]
    DP_REPLICATE_FIELD_NUMBER: _ClassVar[int]
    DP_SHARD_FIELD_NUMBER: _ClassVar[int]
    ENABLE_LOSS_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    PP_FIELD_NUMBER: _ClassVar[int]
    TP_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    cp: int
    dp_replicate: int
    dp_shard: int
    enable_loss_parallel: bool
    pp: int
    tp: int
    world_size: int
    def __init__(self, dp_replicate: _Optional[int] = ..., dp_shard: _Optional[int] = ..., cp: _Optional[int] = ..., tp: _Optional[int] = ..., pp: _Optional[int] = ..., world_size: _Optional[int] = ..., enable_loss_parallel: bool = ...) -> None: ...

class SequenceValue(_message.Message):
    __slots__ = ["elements"]
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedCompositeFieldContainer[NodeValue]
    def __init__(self, elements: _Optional[_Iterable[_Union[NodeValue, _Mapping]]] = ...) -> None: ...

class TensorShape(_message.Message):
    __slots__ = ["dims"]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...

class TraceResult(_message.Message):
    __slots__ = ["graph_module", "parallel_dims"]
    GRAPH_MODULE_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_DIMS_FIELD_NUMBER: _ClassVar[int]
    graph_module: GraphModuleData
    parallel_dims: ParallelConfig
    def __init__(self, parallel_dims: _Optional[_Union[ParallelConfig, _Mapping]] = ..., graph_module: _Optional[_Union[GraphModuleData, _Mapping]] = ...) -> None: ...
