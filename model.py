import gradio as gr
from gpt4all import GPT4All

# GPT4All-Jモデルのロード
chatbot = GPT4All('GPT4All-J')

#  入力と出力の関数を定義
def generate_response(input_text):
    response = chatbot.generate(input_text)
    return response

# Gradioインターフェースの作成
iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")

#  インターフェースの起動
iface.launch(share=True)
