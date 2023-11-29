from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from transformers.generation import GenerationConfig
from peft import PeftModel
import logging
import colorama
import torch
from .base_model import BaseLLMModel
from ..presets import MODEL_METADATA
from threading import Thread


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [250680, 250681, 250682, 50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    

class Huozi_Client(BaseLLMModel):
    def __init__(self, model_name, lora_path=None, user_name="") -> None:
        super().__init__(model_name=model_name, user=user_name)
        self.tokenizer = AutoTokenizer.from_pretrained("models/"+MODEL_METADATA[model_name]["repo_id"])
        self.model = AutoModelForCausalLM.from_pretrained("models/"+MODEL_METADATA[model_name]["repo_id"],
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto")
        if lora_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                "lora/"+lora_path,
                torch_dtype=torch.float16,
            )
        self.max_generation_token = 512
        self.model = self.model.eval()

    def generation_config(self):
        return GenerationConfig.from_dict({
            "temperature": self.temperature,
            # "top_k": 40,
            "top_p": self.top_p,
            # "top_p": 0.75,
            # "num_beams": 4,
            })

    def _get_huozi_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        if self.system_prompt is not None:
            query = self.system_prompt.replace("{instruction}", query)
        logging.debug(colorama.Fore.YELLOW +
                      f"{history}" + colorama.Fore.RESET)
        assert (
            len(history) % 2 == 0
        ), f"History should be even length. current history is: {history}"
        history = [[history[i], history[i + 1]]
                   for i in range(0, len(history), 2)]
        return history, query

    def get_answer_at_once(self):
        history, query = self._get_huozi_style_input()

        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        self.model.generation_config = self.generation_config()

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.model.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )
        response = self.tokenizer.decode(generation_output.sequences[0])
        if "### 回答：" in response:
            response = response.split("### 回答：")[1]
        response = response.split("<|endoftext|>")[0].split("<|beginofutterance|>")[0].split("<|endofutterance|>")[0].strip('\n').strip('</s>')
        return response, len(response)

    def get_answer_stream_iter(self):
        stop = StopOnTokens()
        history, query = self._get_huozi_style_input()

        inputs = self.tokenizer(query, return_tensors="pt").to("cuda")
        self.model.generation_config = self.generation_config()
        streamer = TextIteratorStreamer(self.tokenizer)

        # generation_kwargs = dict(
        #     inputs,
        #     streamer=streamer,
        #     generation_config=self.model.generation_config,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_new_tokens=256,
        # )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.max_generation_token,
            do_sample=True,
            top_p=self.top_p,
            top_k=40,
            temperature=self.temperature,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        response = ""
        for new_text in streamer:
            response += new_text
            if "### 回答：" in response:
                response = response.split("### 回答：")[1]
            yield response.split("<|endoftext|>")[0].split("<|beginofutterance|>")[0].split("<|endofutterance|>")[0].strip('\n').strip('</s>')
