import os
from typing import List, Optional, Tuple, Union

import torch
import xformers.ops.fmha as xformers
from peft import PeftConfig, PeftModel, PeftModelForCausalLM
from transformers import AutoConfig
from transformers import __version__ as transformers_version
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaModel,
    LlamaSdpaAttention,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralFlashAttention2,
    MistralForCausalLM,
    MistralModel,
    MistralSdpaAttention,
)
from unsloth import FastLanguageModel
from unsloth.kernels import fast_cross_entropy_loss
from unsloth.models.gemma import FastGemmaModel
from unsloth.models.llama import (
    FastLlamaModel,
    LlamaAttention_fast_forward,
    LlamaDecoderLayer_fast_forward,
    LlamaLinearScalingRotaryEmbedding,
    LlamaModel_fast_forward,
    LlamaModel_fast_forward_inference,
    LlamaRotaryEmbedding,
    PeftModelForCausalLM_fast_forward,
)
from unsloth.models.loader import SUPPORTS_GEMMA, _get_model_name
from unsloth.models.mistral import FastMistralModel, MistralAttention_fast_forward
from unsloth.models.qwen2 import FastQwen2Model


class FastLanguageModelPatched(FastLanguageModel):
    @staticmethod
    def from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=None,
        dtype=None,
        load_in_4bit=True,
        token=None,
        device_map="sequential",
        rope_scaling=None,
        fix_tokenizer=True,
        trust_remote_code=False,
        use_gradient_checkpointing=True,
        resize_model_vocab=None,
        *args,
        **kwargs,
    ):
        if token is None and "HF_TOKEN" in os.environ:
            token = os.environ["HF_TOKEN"]

        if token is None and "HUGGINGFACE_TOKEN" in os.environ:
            token = os.environ["HUGGINGFACE_TOKEN"]

        old_model_name = model_name
        model_name = _get_model_name(model_name, load_in_4bit)

        # First check if it's a normal model via AutoConfig
        is_peft = False
        try:
            model_config = AutoConfig.from_pretrained(model_name, token=token)
            is_peft = False
        except:
            try:
                # Most likely a PEFT model
                peft_config = PeftConfig.from_pretrained(model_name, token=token)
            except:
                raise RuntimeError(
                    f"Unsloth: `{model_name}` is not a full model or a PEFT model."
                )

            # Check base model again for PEFT
            model_name = _get_model_name(
                peft_config.base_model_name_or_path, load_in_4bit
            )
            model_config = AutoConfig.from_pretrained(model_name, token=token)
            is_peft = True
        pass

        model_type = model_config.model_type

        if model_type == "llama":
            dispatch_model = FastLlamaModelPatched
        elif model_type == "mistral":
            dispatch_model = FastMistralModelPatched
        elif model_type == "gemma":
            if not SUPPORTS_GEMMA:
                raise RuntimeError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma.\n"
                    f"The minimum required version is 4.38.\n"
                    f'Try `pip install --upgrade "transformers>=4.38"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )
            dispatch_model = FastGemmaModel
        elif model_type == "qwen2":
            dispatch_model = FastQwen2Model
        else:
            raise NotImplementedError(
                f"Unsloth: {model_name} not supported yet!\n"
                "Make an issue to https://github.com/unslothai/unsloth!",
            )
        pass

        # Check if this is local model since the tokenizer gets overwritten
        if (
            os.path.exists(os.path.join(old_model_name, "tokenizer_config.json"))
            and os.path.exists(os.path.join(old_model_name, "tokenizer.json"))
            and os.path.exists(os.path.join(old_model_name, "special_tokens_map.json"))
        ):

            tokenizer_name = old_model_name
        else:
            tokenizer_name = None
        pass

        model, tokenizer = dispatch_model.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
            device_map=device_map,
            rope_scaling=rope_scaling,
            fix_tokenizer=fix_tokenizer,
            model_patcher=dispatch_model,
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            *args,
            **kwargs,
        )

        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)

        # In case the model supports tagging, add the unsloth tag.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(
                [
                    "unsloth",
                ]
            )
        pass
        if hasattr(tokenizer, "add_model_tags"):
            tokenizer.add_model_tags(
                [
                    "unsloth",
                ]
            )
        pass

        if load_in_4bit:
            # Fix up bitsandbytes config
            quantization_config = {
                # Sometimes torch_dtype is not a string!!
                "bnb_4bit_compute_dtype": model.config.to_dict()["torch_dtype"],
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_enable_fp32_cpu_offload": False,
                "llm_int8_has_fp16_weight": False,
                "llm_int8_skip_modules": None,
                "llm_int8_threshold": 6.0,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "quant_method": "bitsandbytes",
            }
            model.config.update({"quantization_config": quantization_config})
        pass

        if is_peft:
            # Now add PEFT adapters
            model = PeftModel.from_pretrained(model, old_model_name, token=token)
            # Patch it as well!
            model = dispatch_model.patch_peft_model(model, use_gradient_checkpointing)
        pass
        return model, tokenizer

    pass


pass


def CausalLM_fast_forward_patched(fast_forward_inference):
    def _CausalLM_fast_forward(
        self,
        input_ids: torch.LongTensor = None,
        causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if past_key_values is not None:
            outputs = fast_forward_inference(
                self,
                input_ids,
                past_key_values,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        else:
            causal_mask = xformers.attn_bias.LowerTriangularMask()

            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            self.model._has_no_labels = labels is None

            outputs = self.model(
                input_ids=input_ids,
                causal_mask=causal_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        pass

        hidden_states = outputs[0]
        bsz, q_len, hd = hidden_states.shape
        lm_head = self.lm_head.weight
        if bsz == 1 and q_len == 1:
            logits = torch.mv(lm_head, hidden_states.ravel().to(lm_head.dtype))
            logits = logits.unsqueeze(0).unsqueeze(0)
        else:
            logits = self.lm_head(hidden_states.to(lm_head.dtype))
        pass
        logits = logits.to(self.config.torch_dtype)

        loss = None
        if self.model.training and labels is not None:
            shift_logits = logits
            if not hasattr(self, "extra_ignored_labels"):
                # Fixes https://github.com/unslothai/unsloth/issues/10
                self.extra_ignored_labels = torch.full(
                    (self.max_seq_length, 1), -100, device="cuda"
                )
            pass

            shift_labels = torch.hstack(
                (labels[..., 1:], self.extra_ignored_labels[: labels.shape[0]])
            )
            loss = fast_cross_entropy_loss(
                logits=shift_logits,
                labels=shift_labels,
            )
        elif labels is not None:
            loss = torch.tensor(-1.0)  # loss cannot be directly computed in inference
        pass

        logits = logits[:, -1]  # we only need last position logits for inference

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    pass
    return _CausalLM_fast_forward


pass


class FastLlamaModelPatched(FastLlamaModel):
    @staticmethod
    def custom_pre_patch():
        LlamaAttention.forward = LlamaAttention_fast_forward
        LlamaSdpaAttention.forward = LlamaAttention_fast_forward
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        LlamaDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        LlamaModel.forward = LlamaModel_fast_forward
        LlamaForCausalLM.forward = CausalLM_fast_forward_patched(
            LlamaModel_fast_forward_inference
        )
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.llama.modeling_llama

        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = (
            LlamaLinearScalingRotaryEmbedding
        )
        return

    pass

    FastLlamaModel.pre_patch = custom_pre_patch


def MistralForCausalLM_fast_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if causal_mask is None and past_key_values is None:
        bsz, q_len = input_ids.shape
        sliding_window = getattr(self.config, "sliding_window", None)
        if sliding_window is None or sliding_window == "null" or sliding_window <= 0:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
        elif q_len <= sliding_window:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
        else:
            # Fix from https://github.com/Rypo
            causal_mask = xformers.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                [q_len] * bsz
            ).make_local_attention(window_size=sliding_window)
    pass

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    self.model._has_no_labels = labels is None

    if past_key_values is not None:
        outputs = LlamaModel_fast_forward_inference(
            self,
            input_ids,
            past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    else:
        outputs = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    pass

    hidden_states = outputs[0]
    bsz, q_len, hd = hidden_states.shape
    lm_head = self.lm_head.weight
    if bsz == 1 and q_len == 1:
        logits = torch.mv(lm_head, hidden_states.ravel().to(lm_head.dtype))
        logits = logits.unsqueeze(0).unsqueeze(0)
    else:
        logits = self.lm_head(hidden_states.to(lm_head.dtype))
    pass
    logits = logits.to(self.config.torch_dtype)

    loss = None
    if self.model.training and labels is not None:
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full(
                (self.max_seq_length, 1), -100, device="cuda"
            )
        pass

        shift_labels = torch.hstack(
            (labels[..., 1:], self.extra_ignored_labels[: labels.shape[0]])
        )
        loss = fast_cross_entropy_loss(
            logits=shift_logits,
            labels=shift_labels,
        )
    elif labels is not None:
        loss = torch.tensor(-1.0)  # loss cannot be directly computed in inference
    pass

    logits = logits[:, -1]  # we only need last position logits for inference

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


pass


class FastMistralModelPatched(FastMistralModel):
    @staticmethod
    def custom_pre_patch():
        MistralAttention.forward = MistralAttention_fast_forward
        MistralSdpaAttention.forward = MistralAttention_fast_forward
        MistralFlashAttention2.forward = MistralAttention_fast_forward
        MistralDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        MistralModel.forward = LlamaModel_fast_forward
        MistralForCausalLM.forward = MistralForCausalLM_fast_forward_patched
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.mistral.modeling_mistral

        transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        return

    pass

    FastMistralModel.pre_patch = custom_pre_patch
