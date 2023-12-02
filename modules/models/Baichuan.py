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
    

class Baichuan_Client(BaseLLMModel):
    def __init__(self, model_name, lora_path=None, user_name="") -> None:
        super().__init__(model_name=model_name, user=user_name)
        self.tokenizer = AutoTokenizer.from_pretrained("models/"+MODEL_METADATA[model_name]["repo_id"], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("models/"+MODEL_METADATA[model_name]["repo_id"],
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True)
        # if lora_path is not None:
        #     self.model = PeftModel.from_pretrained(
        #         self.model,
        #         "lora/"+lora_path,
        #         torch_dtype=torch.float16,
        #     )
        self.max_generation_token = 512
        self.model = self.model.eval()
        self.pt_info_dic = {
        'y4251478': '男，40岁，已婚，河北省籍，工人。主因检查发现肝脏占位20天于2019-5-16入院。患者既往十二指肠球炎病史，\
                        定期检查腹部情况，2019年4月16日于当地医院行上腹部增强CT检查提示：肝S5段可见一大小约2.7x2.1cm片状不均匀明显强化。\
                        印象：肝S5段占位，考虑肝癌，请结合临床。后未行特殊治疗。病程中患者无恶心呕吐、腹痛黄疸、肩背部放射性痛等症状。\
                        上腹部增强CT：肝S5段可见一大小约2.7x2.1cm片状不均匀明显强化。印象：肝S5段占位，考虑肝癌，请结合临床。\
                        患者目前肝脏S5段占位，考虑肝癌诊断明确，有手术适应症，限期手术。',
        'y3849370': '男，72岁，已婚，天津南开区。主因体检发现肝癌并门脉癌栓十余天于2018-11-11入院。 二十年前因手术输血感染丙肝病毒,\
                    患者2018年10月23日在天津第一中心医院体检时，B超提示肝脏占位。AFP45.34ng/ml。胆胰脾增强CT提示：肝右叶S6肝癌并门静脉右后支癌栓。\
                    患者进一周出现右上腹部不适，无发热、恶心、呕吐等不适。患者未行特殊处理。2018年10月31日 天津市第一中心医院 肝右叶S6肝癌并门静脉右后支癌栓。 ',
        'y6963130':'女，66岁，汉族，山东济南历城区人，工作单位：无，已婚。主 诉：检查发现肝囊性占位2月余。甲胎蛋白AFP61.70，\
                    现病史：患者自述于2022年6月15日于社区老年查体彩超后发现肝脏囊肿，患者无腹痛腹胀，无恶心呕吐，无畏寒发热，无咳喘。\
                    随后7月份于山东省立医院，行CT,磁共振检查，提示“肝脏囊腺瘤”。为进一步明确诊治到我院就诊，(2022-07-24 本院)行肝脏CTA检查提示：\
                    1、肝脏右叶囊实性病变，恶性不能除外。2、腹主动脉粥样硬化表现。3、肝右动脉起源变异。(2022-07-25本院)行核医学(PET/CT)检查提示：\
                    1.肝右叶团块状囊实性灶,实性部分代谢轻度增高,考虑良性或低度恶性（囊腺瘤恶变）可能性大。2.视野内双肺内多发无代谢磨玻璃结节,\
                    考虑多中心早期肺癌可能性大,建议结合胸部诊断CT；左肺上叶舌段局限性肺气肿；左肺上叶炎症,建议治疗后复查。余腹部PET/CT检查未见明显异常代谢征象。\
                    遂门诊拟“右肝囊腺瘤”收入院。患者病程中无腹泻、无黄疸，无恶心、呕吐，无寒战，高热等症状。',
        'y4257514':'男，77岁，已婚，河北廊坊市籍，。主因检查发现肝脏占位20余天于2019-5-20入院。患者20余天前因脑梗入当地检查治疗，\
                    行腹部CT平扫示：肝左叶肿块，建议增强CT检查；肝硬化。后进一步行腹部平扫+增强MRI提示：肝表面波浪状改变，肝S2段可见类圆形T1WI低、T2WI高信号肿块，大小约3.7cmX3.0cmX3.2cm。\
                    印象：肝S2段富血供肿块，考虑为肝细胞肝癌；脂肪肝、肝硬化表现。后未行特殊治疗。病程中患者偶伴腹胀，上腹部轻微腹痛，不伴发热、恶心呕吐、黄疸黑便等症状。\
                    肝左叶肿块，建议增强CT检查；肝硬化。腹部平扫+增强MRI（中国石油天然气集团公司中心医院 2019-4-29）：\
                    肝表面波浪状改变，肝S2段可见类圆形T1WI低、T2WI高信号肿块，大小约3.7cmX3.0cmX3.2cm。印象：肝S2段富血供肿块，考虑为肝细胞肝癌；脂肪肝、肝硬化表现。',
                    }

    def generation_config(self):
        return GenerationConfig.from_dict({
            "temperature": self.temperature,
            # "top_k": 40,
            "top_p": self.top_p,
            # "top_p": 0.75,
            # "num_beams": 4,
            })
    
    def _get_baichuan_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        logging.debug(colorama.Fore.YELLOW +
                      f"{history}" + colorama.Fore.RESET)
        assert (
            len(history) % 2 == 0
        ), f"History should be even length. current history is: {history}"
        return history, query

    def _get_answer_messages(self):
        history, query = self._get_baichuan_style_input()
        messages = []
        for i in range(0, len(history), 2):
            if history[i] in self.pt_info_dic.keys():
                user_0 = (self.system_prompt or "") + self.pt_info_dic[history[i]]
                messages.append({"role": "user", "content": user_0})
            else:
                messages.append({"role": "user", "content": history[i]})
            messages.append({"role": "assistant", "content": history[i+1]})
        if query in self.pt_info_dic.keys():
            user_0 = (self.system_prompt or "") + self.pt_info_dic[query]
            messages.append({"role": "user", "content": user_0})
        else:
            messages.append({"role": "user", "content": query})
        return messages

    def get_answer_at_once(self):
        history, query = self._get_baichuan_style_input()
        messages = []
        if query in self.pt_info_dic.keys():
            user_0 = self.system_prompt or "" + self.pt_info_dic[query]
            messages.append({"role": "user", "content": user_0})
        else:
            for i in range(0, len(history), 2):
                messages.append({"role": "user", "content": history[i]})
                messages.append({"role": "assistant", "content": history[i+1]})
            messages.append({"role": "user", "content": query})
        messages = self._get_answer_messages()
        response = self.model.chat(self.tokenizer, messages)
        return response, len(response)

    def get_answer_stream_iter(self):
        response = ""
        position = 0
        messages = self._get_answer_messages()
        for i in self.model.chat(self.tokenizer, messages, stream=True):
            response += i[position:]
            position = len(response)
            yield response
