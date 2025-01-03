from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel 
import tensorflow as tf 

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

def read_prompt_and_output(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors = "tf")
    beam_output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output

while True:
    prompt = input("text: ")
    print(read_prompt_and_output(prompt))