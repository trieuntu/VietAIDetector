"""VietAIDetector — Main Entry Point"""

from frontend.gradio_app import create_app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_api=False,
        debug=True,
    )
