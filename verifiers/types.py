from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.completion import Completion
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)

# typing aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Union[Completion, ChatCompletion, None]
ChatMessageField = Literal["role", "content", "tool_calls", "tool_call_id"]
ChatMessage = Dict[ChatMessageField, str | List[ChatCompletionMessageToolCall]]
Message = Union[str, ChatMessage]
Messages = Union[str, List[ChatMessage]]
Info = Dict[str, Any]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
RewardFunc = Callable[..., float]

# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]


class GenerateInputs(TypedDict):
    prompt: List[Messages]
    answer: Optional[List[str]]
    info: Optional[List[Dict]]
    task: Optional[List[str]]
    completion: Optional[List[Messages]]


GenerateOutputs = Dict[str, Any]


class ProcessedOutputs(TypedDict):
    prompt_ids: List[int]
    prompt_mask: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    completion_logprobs: List[float]
    rewards: List[float]
