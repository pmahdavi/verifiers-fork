from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
import random
import time
from typing import List, Dict, Sequence, Any, Union

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from ..imports import LLM, SamplingParams  # type: ignore

from verifiers.envs.environment import Environment


class MultiStepEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 sleep_time: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        #for i, j in enumerate(live_indices):
        def update_state(j, llm_response):
            # sleep for 0-1 seconds to avoid rate limiting
            time.sleep(self.sleep_time * random.random())

            state = states[j].copy()
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})
        
            # get token lengths of env response and new completion
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len # type: ignore
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # update completion masks
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids) # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]

            if self.is_completed(state["messages"]) or len(state["completion_ids"]) > sampling_params.max_tokens: # type: ignore
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:sampling_params.max_tokens]
            else:
                state["messages"].append(self.env_response(state["messages"]))

            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(state["messages"])
                print(state["completion_mask"])
                print(state["completion_ids"])
                raise ValueError(f"Completion mask and completion ids are not the same length for state {j}")

            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
            ))

        for j, state in results:
            states[j] = state

        return states

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [{
            "messages": m,
            "prompt_messages": len(m),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask
        }
        return output

    def eval(self, 
             model: Union[str, LLM],
             batch_size: int = 10, **kwargs: Any):
        if self.eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset()

        rewards = []
        return self.eval_dataset, rewards
    

    