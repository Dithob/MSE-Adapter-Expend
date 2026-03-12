import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class Language_model(nn.Module):
    """
    Drop-in replacement for the original Qwen text model wrapper.

    Expected interface from CMCM:
        - text_embedding(input_ids) -> [B, T, H]
        - forward(llm_input_embeds, labels) -> outputs with .loss
        - generate(llm_input_embeds) -> generated token ids / decoded text
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = args.pretrain_LM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True if str(self.model_name).startswith("/") else False,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": self._resolve_dtype(getattr(args, "llm_dtype", "bf16")),
            "trust_remote_code": True,
        }

        attn_impl = getattr(args, "attn_implementation", None)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        if str(self.model_name).startswith("/"):
            model_kwargs["local_files_only"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        self.hidden_size = self.model.config.hidden_size
        self.embed_tokens = self.model.get_input_embeddings()

        if getattr(args, "freeze_llm", True):
            for p in self.model.parameters():
                p.requires_grad = False

        self.prompt_ids = self._build_prompt_ids(args.task_specific_prompt)

        # classification label mapping, e.g.
        # {'neutral': 0, 'surprise': 1, ...} -> {0: 'neutral', 1: 'surprise', ...}
        self.id2label = None
        if hasattr(args, "label_index_mapping") and args.label_index_mapping is not None:
            # self.id2label = {int(v): str(k) for k, v in args.label_index_mapping.items()}
            self.id2label = None
            if hasattr(args, "label_index_mapping") and args.label_index_mapping is not None:
                self.id2label = {int(v): str(v) for k, v in args.label_index_mapping.items()}

    def _resolve_dtype(self, dtype_name: str):
        dtype_name = str(dtype_name).lower()
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp16":
            return torch.float16
        return torch.float32

    def _build_prompt_ids(self, prompt_text: str) -> torch.Tensor:
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
            return_tensors="pt",
        )["input_ids"][0]
        return prompt_ids.long()

    def text_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _class_ids_to_token_ids(self, class_ids: torch.Tensor, device):
        """
        class_ids: [B]
        return: token ids [B, T]
        """
        if self.id2label is None:
            raise ValueError("1D labels detected, but args.label_index_mapping is missing.")

        label_texts = [self.id2label[int(x)] for x in class_ids.tolist()]

        if getattr(self.args, "append_eos_to_label", True) and self.tokenizer.eos_token is not None:
            label_texts = [txt + self.tokenizer.eos_token for txt in label_texts]

        tokenized = self.tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return tokenized["input_ids"].to(device)

    def _normalize_label_ids(self, labels, device):
        """
        Supported:
            - Tensor[B]       : class ids for classification
            - Tensor[B, T]    : token ids
            - tuple/list      : first element is one of above
            - dict            : contains input_ids / labels
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

        label_ids = label_ids.to(device)

        # case 1: classification labels, shape [B]
        if label_ids.ndim == 1:
            return self._class_ids_to_token_ids(label_ids.long(), device)

        # case 2: already token ids, shape [B, T]
        if label_ids.ndim != 2:
            raise ValueError(f"Unsupported label shape: {tuple(label_ids.shape)}")

        label_ids = label_ids.long()

        # replace negative ids with pad token
        label_ids = torch.where(
            label_ids < 0,
            torch.full_like(label_ids, self.tokenizer.pad_token_id),
            label_ids
        )

        # append eos only if needed
        if getattr(self.args, "append_eos_to_label", True):
            last_is_eos = (label_ids[:, -1] == self.tokenizer.eos_token_id)
            if not torch.all(last_is_eos):
                eos_col = torch.full(
                    (label_ids.size(0), 1),
                    self.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device
                )
                label_ids = torch.cat([label_ids, eos_col], dim=1)

        return label_ids

    def _build_prompt_embeds(self, batch_size: int, device) -> torch.Tensor:
        prompt_ids = self.prompt_ids.to(device).unsqueeze(0).expand(batch_size, -1)
        return self.embed_tokens(prompt_ids)

    def forward(self, llm_input_embeds: torch.Tensor, labels):
        """
        llm_input_embeds: [B, T_prefix, H]
        labels:
            - classification ids [B]
            - or token ids [B, T]
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

        # 只取新增 token
        new_tokens = generated[:, -getattr(self.args, "max_new_tokens", 4):]

        decoded = self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )

        return decoded