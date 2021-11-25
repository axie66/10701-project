import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

assert model.config.is_encoder_decoder
x = torch.rand(10, 10, 512)
input_ids = torch.ones(10, 10)
attention_mask = torch.ones(10, 10)

out = model.generate(input_ids=input_ids, inputs_embeds=x, 
                     attention_mask=attention_mask, continuous_prompt=True)