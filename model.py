import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def generate_response(input_text):
    #  入力テキストをトークナイザーでエンコード
    inputs = tokenizer.encode_plus(input_text, return_tensors="pt")
    #  モデルに入力を渡し、答えを生成
    outputs = model(**inputs)
    # 答えのスコアと開始位置を取得
    answer_scores = outputs.start_logits
    answer_starts = torch.argmax(answer_scores)  # get the most likely beginning of answer with the argmax of the score
    # 答えの終了位置を計算
    answer_ends = answer_starts + torch.argmax(outputs.end_logits[answer_starts]) +  1  # get the most likely end of answer with the argmax of the score
    # 答えをトークナイザーでデコード
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_starts:answer_ends]))
    return answer

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")
iface.launch(share=True)