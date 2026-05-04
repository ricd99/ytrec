"""
FASTAPI + GRADIO SERVING APPLICATION

This application provides a complete serving solution for the ytrec model
with both programmatic API access and a user-friendly web interface.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.inference import predict
import gradio as gr

app = FastAPI(
    title="ytrec prediction API",
    description="ML API for showing related long-form yt channels based on a query channel",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"status": "ok"}


class ChannelName(BaseModel):
    channel_name: str


@app.post("/predict")
def get_prediction(channel: ChannelName):
    try:
        result = predict(channel.channel_name)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


def gradio_predict(channel_name):
    try:
        result = predict(channel_name)
        if isinstance(result, dict) and "error" in result:
            return f"<p style='color:red'>{result['error']}</p>"
        if not result:
            return "<p>No channels found.</p>"

        html = """
        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-top:10px;">
        """

        for idx, r in enumerate(result, 1):
            channel_name = r['channel_name']
            channel_id = r['channel_id']
            suffix = "st" if idx == 1 else "nd" if idx == 2 else "rd" if idx == 3 else "th"

            if idx == 1:
                bg, text_color, link_color = "#FFD700", "#333333", "#003366"
            elif idx == 2:
                bg, text_color, link_color = "#C0C0C0", "#333333", "#003366"
            elif idx == 3:
                bg, text_color, link_color = "#CD7F32", "#ffffff", "#ffe0b2"
            else:
                bg, text_color, link_color = "#2a2a2a", "#ffffff", "#ff9933"

            html += f"""
            <div style="border:1px solid #333; border-radius:8px; padding:16px; background:{bg}; margin-bottom:8px;">
                <p style="font-size:0.9em; color:{text_color}; margin-bottom:8px; font-weight:bold;">{idx}{suffix} closest match</p>
                <h3 style="margin:0 0 8px 0; color:{text_color};">{channel_name}</h3>
                <a href="https://www.youtube.com/channel/{channel_id}" target="_blank" style="color:{link_color}; text-decoration:none; font-weight:bold;">
                    Watch on YouTube →
                </a>
            </div>
            """

        html += "</div>"
        return html
    except Exception as e:
        return f"<p style='color:red'>error: {str(e)}</p>"


css = "body { background-color: #000000 !important; color: #ffffff !important; }"

with gr.Blocks(title="similar channels to watch when you eat", css=css) as demo:
    gr.Markdown("# Similar Channels to Watch While You Eat")
    gr.Markdown(
        "Can't find a youtube channel to watch and your food is getting cold? "
        "Enter a name of a youtube channel you enjoyed while eating recently."
    )

    with gr.Row():
        with gr.Column(scale=1):
            channel_input = gr.Textbox(
                label="Channel Name",
                placeholder="e.g., fern, Secret Base",
                lines=1
            )
            search_btn = gr.Button("Find Similar Channels", variant="primary")
            gr.Markdown("### Examples")
            gr.Examples(
                examples=[["LEMMiNO"], ["Secret Base"]],
                inputs=channel_input,
            )

        with gr.Column(scale=2):
            output_html = gr.HTML(
                value="<p style='color:#aaa; text-align:center; padding:40px; font-size:1.1em;'>"
                       "Enter a channel name and click search to see recommendations.</p>",
                label="Recommended Channels"
            )

    search_btn.click(fn=gradio_predict, inputs=channel_input, outputs=output_html)
    channel_input.submit(fn=gradio_predict, inputs=channel_input, outputs=output_html)

app = gr.mount_gradio_app(app, demo, path="/ui")
