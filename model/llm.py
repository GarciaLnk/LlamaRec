from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import (
    Cache,
    CausalLMOutputWithPast,
    LlamaForCausalLM,
)
from unsloth.models.llama import (
    CausalLMOutputWithPast,
    FastLlamaModel,
    LlamaAttention,
    LlamaAttention_fast_forward,
    LlamaDecoderLayer,
    LlamaDecoderLayer_fast_forward,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaLinearScalingRotaryEmbedding,
    LlamaModel,
    LlamaModel_fast_forward,
    LlamaModel_fast_forward_inference,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
    PeftModelForCausalLM,
    PeftModelForCausalLM_fast_forward,
    fast_cross_entropy_loss,
    xformers,
)


class LlamaForCausalLMPatched(LlamaForCausalLM):
    def forward_patched(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if self.training and labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        elif labels is not None:
            loss = torch.tensor(-1.0)  # loss cannot be directly computed in inference

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

    LlamaForCausalLM.forward = forward_patched


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
