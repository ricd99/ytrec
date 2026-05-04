"""
FASTAPI + (GRADIO SERVING APPLICATION - soon!)

This application provides a complete serving solution for the ytrec model
with both programmatic API access and (a user-friendly web interface - soon!).

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- (Gradio: User-friendly web UI for manual testing and demonstrations - soon!)
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.inference import predict
import gradio as gr

app = FastAPI(
    title = "ytrec prediction API",
    description = "ML API for showing related long-form yt channels based on a query channel",
    version = "1.0.0"
)


@app.get("/")
def root():
    """
    Health check endpoint for AWS Application Load Balancer
    """
    return {"status": "ok"}


class ChannelName(BaseModel):
    """
    Yt channel data schema
    """
    channel_name: str


@app.post("/predict")
def get_prediction(channel: ChannelName):
    try:
        result = predict(channel.channel_name)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    



# === GRADIO WEB INTERFACE ===
def gradio_predict(channel_name):
    """
    Gradio interface function that processes form inputs and returns prediction.

    This function:
    1. Takes individual form inputs from Gradio UI
    2. Calls the same inference pipeline used by the API
    3. Returns formatted prediction string

    """

    try:
        result = predict(channel_name)
        if isinstance(result, dict) and "error" in result:
            return result["error"]

        output = ""
        for r in result:
            output += f"{r['channel_name']}\n"
            output += f"https://www.youtube.com/channel/{r['channel_id']}\n"
            output += f"similarity score: {r['similarity_score']:.4f}\n\n"
        return output.strip()
    except Exception as e:
        return "error: " + str(e)


# === GRADIO UI CONFIGURATION ===
# Using gr.Blocks for flexible layout structure with rows and columns
with gr.Blocks(title="similar channels to watch when you eat") as demo:
    gr.Markdown("# 🍽️ Similar Channels to Watch While You Eat")
    gr.Markdown(
        "Can't find a yt channel to watch and your food is getting cold? "
        "Enter a name of a youtube channel you enjoyed while eating recently."
    )

    with gr.Row():
        with gr.Column(scale=1):
            channel_input = gr.Textbox(
                label="Channel Name",
                placeholder="e.g., fern, Secret Base",
                lines=1
            )
            search_btn = gr.Button("🔍 Find Similar Channels", variant="primary")

            gr.Markdown("### Examples")
            gr.Examples(
                examples=[["fern"], ["Secret Base"]],
                inputs=channel_input,
            )

        with gr.Column(scale=2):
            output_box = gr.Textbox(
                label="Recommended YT Channels",
                lines=10,
                interactive=False
            )

    # Connect the button click to the prediction function
    search_btn.click(
        fn=gradio_predict,
        inputs=channel_input,
        outputs=output_box
    )

    # Also allow Enter key to trigger search
    channel_input.submit(
        fn=gradio_predict,
        inputs=channel_input,
        outputs=output_box
    )

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)