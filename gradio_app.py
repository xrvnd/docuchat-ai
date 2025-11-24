import gradio as gr
import uuid
from pathlib import Path
import shutil

from main import extract_text_from_file, search_documents, ve

def upload_file(files):
    if not files:
        return "No file uploaded."

    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)

    count = 0

    for f in files:
        src = Path(f.name)
        dst = uploads / src.name
        shutil.copy(src, dst)

        try:
            text = extract_text_from_file(str(dst))
            doc_id = f"doc_{uuid.uuid4()}"
            ve.add_document(doc_id, text, {"filename": src.name})
            count += 1
        except Exception as e:
            return f"❌ Error on {src.name}: {e}"

    return f"✅ {count} documents uploaded. Total: {ve.count()}"

def chat(history, user_input):
    results = search_documents(user_input)
    docs = results["documents"][0]

    if not docs:
        reply = "No relevant documents found."
    else:
        reply = ve.generate_answer(user_input, docs)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

    return history, ""

def get_stats():
    return {"documents": ve.count()}

with gr.Blocks(title="RAG Chat") as demo:
    with gr.Tab("Chat"):
        chatbox = gr.Chatbot(
            label="Chat with RAG",
            height=600,
            type="messages"
        )
        txt = gr.Textbox()
        btn = gr.Button("Send")
        btn.click(chat, [chatbox, txt], [chatbox, txt])

    with gr.Tab("Upload"):
        files = gr.File(file_count="multiple")
        up_btn = gr.Button("Upload")
        out = gr.Textbox()
        up_btn.click(upload_file, [files], out)

    with gr.Tab("Stats"):
        stats_btn = gr.Button("Refresh")
        stats_box = gr.JSON()
        stats_btn.click(get_stats, None, stats_box)

demo.launch(server_name="0.0.0.0", server_port=2001)
