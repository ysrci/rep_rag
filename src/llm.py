from typing import Any
from .settings import Settings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Small_LLM:
    """Wrapper around a Hugging Face causal language model for generation."""

    def __init__(self, model: str = Settings.LLM_MODEL):
        """Initialize the tokenizer and causal language model.

        Args:
            model (str): Hugging Face model identifier or checkpoint path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch.float32 if not torch.cuda.is_available() else "auto",
            device_map="auto"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = Settings.MAX_NEW_TOKENS
    ) -> str:
        """Generate a text completion for a prompt using chat formatting.

        Args:
            prompt (str): Prompt text to provide to the model.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: Generated text response.
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        response = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )

        return str(response)
