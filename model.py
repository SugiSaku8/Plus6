import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-13b")
model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-13b")

def generate_response(input_text):
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")
iface.launch(share=True)
