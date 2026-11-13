"""
RedBus AI Assistant – Intelligent Bus Ticket Booking Automation
Module 1: Gradio Frontend UI Development

This module develops the frontend chatbot interface using Python and Gradio.
It features RedBus travel-themed styling, session chat history, quick suggestion buttons,
and beginner-friendly rule-based responses.
"""

import os
import sys
import time
import gradio as gr

# Ensure modules package is discoverable even when run from different directories
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from modules.chat_logic import get_bot_response, WELCOME_MESSAGE

# Custom CSS for RedBus Travel Theme (Red, White, and Light Grey palette)
CUSTOM_CSS = """
/* Global Container & Page Styling */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    background-color: #F8F9FA !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Header Banner Card */
.header-card {
    background: linear-gradient(135deg, #D84E55 0%, #B8353C 100%);
    border-radius: 12px;
    padding: 24px 30px;
    margin-bottom: 20px;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(216, 78, 85, 0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-content h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}

.header-content p {
    font-size: 14px !important;
    margin: 6px 0 0 0 !important;
    color: #FDE8E9 !important;
    opacity: 0.95;
}

.header-badge {
    background-color: rgba(255, 255, 255, 0.2);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Main Chat Card */
.chat-card {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 16px !important;
}

/* Chatbot Window & Bubbles */
.chat-window {
    border-radius: 12px !important;
    border: 1px solid #ECEEF1 !important;
    background-color: #FFFFFF !important;
}

/* Quick Suggestions Bar */
.suggestions-label {
    font-size: 13px;
    font-weight: 600;
    color: #555A64;
    margin-bottom: 6px;
}

.suggestion-btn {
    background-color: #FFFFFF !important;
    color: #D84E55 !important;
    border: 1px solid #F0C4C7 !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.suggestion-btn:hover {
    background-color: #FDE8E9 !important;
    border-color: #D84E55 !important;
    box-shadow: 0 2px 6px rgba(216, 78, 85, 0.15) !important;
}

/* Primary Action Buttons */
.redbus-send-btn {
    background-color: #D84E55 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease !important;
}

.redbus-send-btn:hover {
    background-color: #BA3E44 !important;
}

.clear-btn {
    background-color: #F1F3F5 !important;
    color: #495057 !important;
    border: 1px solid #DEE2E6 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.clear-btn:hover {
    background-color: #E9ECEF !important;
    color: #212529 !important;
}

/* Input Textbox */
.input-textbox textarea {
    border-radius: 8px !important;
    border: 1px solid #D0D5DD !important;
    font-size: 14px !important;
}

.input-textbox textarea:focus {
    border-color: #D84E55 !important;
    box-shadow: 0 0 0 2px rgba(216, 78, 85, 0.2) !important;
}

/* Footer note */
.footer-text {
    text-align: center;
    font-size: 12px;
    color: #8C929D;
    margin-top: 10px;
}
"""

def create_initial_history():
    """Returns the starting chat history with the initial welcome message."""
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def bot_respond(user_input, history):
    """
    Handle user input and yield the streaming bot response with typing indicator effect.
    
    Args:
        user_input (str): Message submitted by user.
        history (list): Current conversation history.
        
    Yields:
        tuple: (cleared_input, updated_history)
    """
    if not user_input or not user_input.strip():
        yield "", history
        return

    user_msg = user_input.strip()

    # Append user message
    new_history = list(history) if history else []
    new_history.append({"role": "user", "content": user_msg})

    # Prepare placeholder for bot response
    new_history.append({"role": "assistant", "content": ""})

    # Generate rule-based reply
    bot_reply = get_bot_response(user_msg)

    # Word-by-word streaming effect to provide a natural typing indicator
    words = bot_reply.split(" ")
    accumulated_text = ""
    for idx, word in enumerate(words):
        accumulated_text += (word + " " if idx < len(words) - 1 else word)
        new_history[-1]["content"] = accumulated_text
        time.sleep(0.02)  # subtle natural pacing
        yield "", new_history


def apply_suggestion(suggestion_text, history):
    """
    Directly execute a quick suggestion button click and stream the bot response.
    
    Args:
        suggestion_text (str): Clicked prompt text.
        history (list): Current conversation history.
        
    Yields:
        tuple: (cleared_input, updated_history)
    """
    yield from bot_respond(suggestion_text, history)


def clear_conversation():
    """
    Reset the chatbot conversation to initial welcome state and clear input field.
    
    Returns:
        tuple: ("", reset_history)
    """
    return "", create_initial_history()


def build_app():
    """
    Build and configure the Gradio Blocks UI layout.
    
    Returns:
        gr.Blocks: Configured Gradio interface.
    """
    with gr.Blocks(title="RedBus AI Assistant") as demo:
        # Inject Custom CSS directly for cross-version Gradio compatibility
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # 1. Navigation / Application Header
        gr.HTML(
            """
            <div class="header-card">
                <div class="header-content">
                    <h1>🚌 RedBus AI Assistant</h1>
                    <p>Your Intelligent Bus Travel Companion</p>
                </div>
                <div class="header-badge">Module 1 • UI Ready</div>
            </div>
            """
        )

        with gr.Column(elem_classes=["chat-card"]):
            # 2. Chatbot Display Window
            # Compatible with Gradio 5 & 6
            try:
                chatbot = gr.Chatbot(
                    value=create_initial_history(),
                    height=420,
                    layout="bubble",
                    elem_classes=["chat-window"],
                    show_label=False
                )
            except TypeError:
                chatbot = gr.Chatbot(
                    value=create_initial_history(),
                    height=420,
                    elem_classes=["chat-window"],
                    show_label=False
                )

            # 3. Quick Suggestion Buttons
            gr.HTML('<div class="suggestions-label">💡 Quick Suggestions:</div>')
            with gr.Row():
                btn_suggest_1 = gr.Button("Chennai to Bangalore", elem_classes=["suggestion-btn"], size="sm")
                btn_suggest_2 = gr.Button("Vijayawada to Hyderabad", elem_classes=["suggestion-btn"], size="sm")
                btn_suggest_3 = gr.Button("Find buses under ₹1000", elem_classes=["suggestion-btn"], size="sm")
                btn_suggest_4 = gr.Button("AC Sleeper Buses", elem_classes=["suggestion-btn"], size="sm")

            # 4. User Input Row (Textbox + Send + Clear)
            with gr.Row():
                user_msg_input = gr.Textbox(
                    placeholder="Type your travel inquiry here (e.g., Chennai to Bangalore, under 1000 rupees)...",
                    show_label=False,
                    scale=7,
                    elem_classes=["input-textbox"],
                    container=False,
                    autofocus=True
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1, elem_classes=["redbus-send-btn"])
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", scale=1, elem_classes=["clear-btn"])

            # 5. Footer Info
            gr.HTML('<div class="footer-text">RedBus AI Assistant • Module 1 (Gradio UI Frontend) • Powered by Python & Gradio</div>')

        # 6. Event Wiring
        # Send on button click or Enter key
        send_btn.click(
            fn=bot_respond,
            inputs=[user_msg_input, chatbot],
            outputs=[user_msg_input, chatbot],
            show_progress="minimal"
        )
        
        user_msg_input.submit(
            fn=bot_respond,
            inputs=[user_msg_input, chatbot],
            outputs=[user_msg_input, chatbot],
            show_progress="minimal"
        )

        # Clear Chat button click
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[user_msg_input, chatbot],
            show_progress=False
        )

        # Quick Suggestion Button Clicks (Directly submits to chatbot)
        btn_suggest_1.click(
            fn=lambda h: apply_suggestion("Chennai to Bangalore", h),
            inputs=[chatbot],
            outputs=[user_msg_input, chatbot],
            show_progress="minimal"
        )

        btn_suggest_2.click(
            fn=lambda h: apply_suggestion("Vijayawada to Hyderabad", h),
            inputs=[chatbot],
            outputs=[user_msg_input, chatbot],
            show_progress="minimal"
        )

        btn_suggest_3.click(
            fn=lambda h: apply_suggestion("Find buses under ₹1000", h),
            inputs=[chatbot],
            outputs=[user_msg_input, chatbot],
            show_progress="minimal"
        )

        btn_suggest_4.click(
            fn=lambda h: apply_suggestion("AC Sleeper Buses", h),
            inputs=[chatbot],
            outputs=[user_msg_input, chatbot],
            show_progress="minimal"
        )

    return demo


if __name__ == "__main__":
    # Launch the Gradio app locally
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Default(primary_hue="red"),
        share=False
    )
