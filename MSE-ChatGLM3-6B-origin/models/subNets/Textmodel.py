import os
import sys
import collections
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ChatGLM3.modeling_chatglm import ChatGLMForConditionalGeneration
from models.ChatGLM3.tokenization_chatglm import ChatGLMTokenizer

__all__ = ['Language_model']

class Language_model(nn.Module):
    def __init__(self, args, use_PLM = True):
        """
        language: en / cn
        """
        super(Language_model, self).__init__()

        if use_PLM:
            pretrained_model = args.pretrain_LM              #pretrained model select
            
            # ====== 修改的核心部分开始：开启极低内存流式加载 ======
            self.model = ChatGLMForConditionalGeneration.from_pretrained(
                pretrained_model, 
                trust_remote_code=True, 
                torch_dtype=torch.float16,   # 直接以半精度加载，省去后续.half()转换带来的内存峰值
                low_cpu_mem_usage=True,      # 🌟 开启低系统内存模式
                device_map="auto"            # 自动分配到 GPU
            )
            # ====== 修改的核心部分结束 ======
            
            self.tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
            self.device = args.device
            self.language = args.language
            self.max_new_tokens = args.max_new_tokens
            self.datasetName = args.datasetName
            self.train_mode = args.train_mode
            self.task_specific_prompt = args.task_specific_prompt
            # freeze parameter
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('please use PLM')

    def text_embedding(self,text_ids):
        embeddings = self.model.base_model.get_input_embeddings()
        return embeddings(text_ids)

    def forward(self, fusion_embedding, labels):
        """
        Args:
            fusion_embedding: the "concatenate" result of  multimodal low rank fusion  and text embedding
            label: ground_truth
        """
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)  #添加多模态输入的special prompt
        opt_tokens, labels = self.input_processing(fusion_embedding, labels, mode = 'train')          #创建fusion+prompt+answer_mask的input和label

        with torch.cuda.amp.autocast():
            output = self.model(input_ids = opt_tokens, input_fusion=fusion_embedding, labels = labels)  # Models outputs are now tuples

        return output

    def generate(self, fusion_embedding):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
            ...
        """
        if self.train_mode == 'regression':
            gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 1, "do_sample": False, "top_k": 10}
        else:
            gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 1, "do_sample": False, "top_k": 10 }

        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)  # 添加多模态输入的special prompt
        opt_tokens, _ = self.input_processing(fusion_embedding, mode = 'generate')  # 创建fusion+prompt的input

        context_length = opt_tokens.size(1)
        all_responses =[]

        for outputs in self.model.stream_generate(opt_tokens, **gen_kwargs, input_fusion=fusion_embedding):
            outputs = outputs[:, context_length:].tolist()
            response = self.tokenizer.batch_decode(outputs)
            
        for x in response:
            if self.train_mode == 'regression':
                try:
                    value = float(
                        x.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(
                            ':', ''))
                except ValueError:
                    value = 0.0
            else:
                try:
                    value = float(x)
                except ValueError:
                    value = 0.0
            all_responses.append(value)
        return all_responses

    def input_processing(self, fusion_embedding, labels = None, mode = None):
        """
        Args:
            fusion_embedding: the "concatenate" result of  multimodal low rank fusion  and text embedding
            ...
        """
        input_lengths = fusion_embedding[:, :, 0]
        fusion_empty =(torch.ones(input_lengths.size(), dtype=torch.long).to(self.device).fill_(0))

        task_prompt = self.get_task_prompt()
        prompt_broadcasted = task_prompt.expand(fusion_empty.size(0), -1)

        opt_tokens = torch.cat([fusion_empty, prompt_broadcasted], dim=1)    #构建fusion+prompt的tokens(其中fusion部分是空白，用于后面填充)

        opt_tokens, labels = self.input_labels_construct(opt_tokens, labels, mode)

        return opt_tokens, labels

    def input_labels_construct(self, opt_tokens, labels = None, mode = None):
        """
        Args:
            opt_tokens: the "concatenate" size of  multimodal low rank fusion, text embedding and prompt
            ...
        """
        batch_size = opt_tokens.shape[0]

        if mode == "train":
            if self.train_mode == "regression":
                label_template = [f"{label.item():.{1}f}" for label in labels]
            else:
                label_template = [f"{label.item()}" for label in labels]

            labels_id = self.tokenizer(label_template, padding=True, return_tensors="pt", add_special_tokens=False)[
                "input_ids"].to(self.device)

            labels_matrix = torch.empty_like(opt_tokens).fill_(-100).long().to(self.device)  # bz * seq_len
            opt_tokens = torch.cat([opt_tokens, labels_id], dim=1)  # 将输入与labels拼接
            labels = torch.cat([labels_matrix, labels_id], dim=1)

        return opt_tokens, labels

    def get_task_prompt(self):
        # get the task_specific_prompt
        prompt_text = self.task_specific_prompt
        prompt_ids = self.tokenizer(prompt_text, padding=True, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

        return prompt_ids

    def multimodal_prompt_wrap(self,fusion_embeddings):
        """
        Args:
            Wrap the input with a special token
        """
        if self.language == "en":
            prompt = '{question}\n\n <Multimodal><MultimodalHere></Multimodal>'
            special_token = '<MultimodalHere>'
        else:
            prompt = '{问题}\n\n <多模态><MultimodalHere></多模态>'
            special_token = '<MultimodalHere>'

        batch_size = fusion_embeddings.shape[0]
        p_before, p_after = prompt.split(special_token)
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", add_special_tokens=True).to(self.device)
        p_after_tokens = self.tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.text_embedding(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.text_embedding(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_fusion_embeddings = torch.cat([p_before_embeds, fusion_embeddings, p_after_embeds], dim=1)

        return wrapped_fusion_embeddings