# models/subNets/Textmodel.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class Language_model(nn.Module):
    """
    A drop-in replacement for the original Qwen-1.8B text model wrapper.

    Expected interface from CMCM:
        - text_embedding(input_ids) -> [B, T, H]
        - forward(llm_input_embeds, labels) -> outputs with .loss
        - generate(llm_input_embeds) -> generated token ids / decoded text

    llm_input_embeds:
        [B, P + T_text, H]
        where P is pseudo token count from MSE-Adapter fusion,
        and T_text is the text token sequence length.

    labels:
        usually [B, T_label] token ids of target answer
        or a tuple/list whose first element is token ids
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = args.pretrain_LM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="right",
            use_fast=False,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = self._resolve_dtype(getattr(args, "llm_dtype", "bf16"))

        model_kwargs = {
            "torch_dtype": torch_dtype,
        }
        attn_impl = getattr(args, "attn_implementation", None)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        self.hidden_size = self.model.config.hidden_size
        self.embed_tokens = self.model.get_input_embeddings()

        # Keep original MSE-Adapter spirit: freeze the backbone by default.
        if getattr(args, "freeze_llm", True):
            for p in self.model.parameters():
                p.requires_grad = False

        # cache prompt ids
        self.prompt_ids = self._build_prompt_ids(args.task_specific_prompt)

    def _resolve_dtype(self, dtype_name: str):
        dtype_name = str(dtype_name).lower()
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp16":
            return torch.float16
        return torch.float32

    def _build_prompt_ids(self, prompt_text: str) -> torch.Tensor:
        """
        Prefer official chat template for Qwen3.5-Base, fallback to raw text tokenization.
        """
        use_chat_template = getattr(self.args, "use_chat_template", True)

        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt_text}]
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if prompt_ids.ndim == 2:
                    prompt_ids = prompt_ids[0]
                return prompt_ids.long()
            except Exception:
                pass

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]
        return prompt_ids.long()

    def text_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        return:    [B, T, H]
        """
        return self.embed_tokens(input_ids)

    def _normalize_label_ids(self, labels, device):
        """
        Try to be compatible with existing dataloader outputs.
        Supported:
            - Tensor[B, T]
            - tuple/list with first element Tensor[B, T]
            - dict containing 'input_ids' or 'labels'
        """
        if isinstance(labels, torch.Tensor):
            label_ids = labels
        elif isinstance(labels, (tuple, list)):
            label_ids = labels[0]
        elif isinstance(labels, dict):
            if "input_ids" in labels:
                label_ids = labels["input_ids"]
            elif "labels" in labels:
                label_ids = labels["labels"]
            else:
                raise ValueError("Unsupported label dict format.")
        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")

        label_ids = label_ids.to(device).long()

        # remove invalid negative ids except ignore_index
        label_ids = torch.where(
            label_ids < 0,
            torch.full_like(label_ids, self.tokenizer.pad_token_id),
            label_ids
        )

        if getattr(self.args, "append_eos_to_label", True):
            eos = torch.full(
                (label_ids.size(0), 1),
                self.tokenizer.eos_token_id,
                dtype=torch.long,
                device=device
            )
            # avoid double append if already ends with eos
            need_append = (label_ids[:, -1] != self.tokenizer.eos_token_id).unsqueeze(1)
            label_ids = torch.cat(
                [label_ids, eos * need_append + label_ids[:, -1:].clone() * (~need_append)],
                dim=1
            )
            # 上面写法会在已带 eos 时复制最后一个 token，不够干净，下面修正：
            for i in range(label_ids.size(0)):
                if label_ids[i, -2] == self.tokenizer.eos_token_id:
                    label_ids[i, -1] = self.tokenizer.eos_token_id

        return label_ids

    def _build_prompt_embeds(self, batch_size: int, device) -> torch.Tensor:
        prompt_ids = self.prompt_ids.to(device).unsqueeze(0).expand(batch_size, -1)
        return self.embed_tokens(prompt_ids)

    def forward(self, llm_input_embeds: torch.Tensor, labels):
        """
        llm_input_embeds: [B, T_prefix, H]
        labels: target token ids, shape [B, T_label]
        """
        device = llm_input_embeds.device
        batch_size = llm_input_embeds.size(0)

        label_ids = self._normalize_label_ids(labels, device=device)
        label_embeds = self.embed_tokens(label_ids)
        prompt_embeds = self._build_prompt_embeds(batch_size, device)

        # prefix = fusion pseudo tokens + original text embeddings + task prompt
        full_inputs_embeds = torch.cat(
            [llm_input_embeds, prompt_embeds, label_embeds],
            dim=1
        )

        ignore_prefix = torch.full(
            (batch_size, llm_input_embeds.size(1) + prompt_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=device
        )
        full_labels = torch.cat([ignore_prefix, label_ids], dim=1)

        attention_mask = torch.ones(
            full_inputs_embeds.size()[:2],
            dtype=torch.long,
            device=device
        )

        outputs = self.model(
            inputs_embeds=full_inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def generate(self, llm_input_embeds: torch.Tensor):
        """
        Return generated ids and decoded strings.
        """
        device = llm_input_embeds.device
        batch_size = llm_input_embeds.size(0)
        prompt_embeds = self._build_prompt_embeds(batch_size, device)

        full_inputs_embeds = torch.cat([llm_input_embeds, prompt_embeds], dim=1)
        attention_mask = torch.ones(
            full_inputs_embeds.size()[:2],
            dtype=torch.long,
            device=device
        )

        generated = self.model.generate(
            inputs_embeds=full_inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=getattr(self.args, "max_new_tokens", 4),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )
        return {
            "output_ids": generated,
            "output_text": decoded,
        }