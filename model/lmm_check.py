import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from topic_classifiier import topic_classify

topic = ""
last_response = ""
model_name = "google/flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
conversation_history=[]

def llm(input_ids):
    generate = model.generate(input_ids,
                                 max_length = 500,
                                 temperature =0.74,
                                 num_beams=2,
                                 top_p =0.9,
                                 repetition_penalty=1.5,
                                 do_sample =True
                                 )
    return generate

def prompt_input(prompt, max_history_length=3):
    global conversation_history
    global topic
    global last_response
    global last_topic

    if topic == "":
        topic = topic_classify(prompt)
        last_topic = topic
    else:
        topic = topic_classify(prompt)
        if topic != last_topic:
            conversation_history=[]
        last_topic = topic

    conversation_history.append(f"\nUser: {prompt}")
    truncated_history =conversation_history[-max_history_length:]
    conversetion_prompt = "\n".join(truncated_history)+"\nBot: "

    input_ids = tokenizer.encode(conversetion_prompt, return_tensors = "pt",truncation=True)
    
    generate = llm(input_ids)

    output = tokenizer.decode(generate[0], skip_special_tokens=True)
    response = output.split("Bot: ")[-1].strip()

    if len(conversation_history) > 1:
        if response == last_response:
            fresh_prompt = f"User: {prompt}\nBot: "
            input_ids = tokenizer.encode(fresh_prompt, return_tensors="pt", truncation=True)
            generate = llm(input_ids)
            output = tokenizer.decode(generate[0], skip_special_tokens=True)
            response = output.split("Bot: ")[-1].strip()

    last_response = response
    conversation_history.append(f"\nBot: {response}")
    
    return response