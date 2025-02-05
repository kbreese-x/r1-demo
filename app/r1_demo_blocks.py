from dataclasses import replace
import gradio as gr
from gradio import ChatMessage
from sagemaker_utils import *


def clear_history():
    return []


def user_message(message: str, history: list[ChatMessage]):
    return "", history + [ChatMessage(role="user", content=message)]


def process_llm_stream(chat_history):
    thinking = ""
    answer = ""
    buffer = ""
    in_think = False

    thinking_message = ChatMessage(
        role="assistant",
        content="",
        metadata={"title": "🤔 Thinking", "status": "pending"},
    )

    for chunk in invoke_endpoint(chat_history):
        buffer += chunk

        if not in_think and "<think>" in buffer:
            in_think = True
            buffer = buffer[buffer.index("<think>") + 7 :]

        if in_think and "</think>" in buffer:
            think_end = buffer.index("</think>")
            thinking += buffer[:think_end]
            answer += buffer[think_end + 8 :]
            in_think = False
            buffer = ""
            # Start yielding answer message once thinking is complete
            yield chat_history + [
                replace(
                    thinking_message,
                    content=thinking,
                    # metadata={"title": "🤔 Thinking", "status": "done"},
                ),
                ChatMessage(role="assistant", content=answer),
            ]
        elif in_think:
            thinking += buffer
            buffer = ""
            yield chat_history + [replace(thinking_message, content=thinking)]
        else:
            answer += buffer
            buffer = ""
            yield chat_history + [
                replace(
                    thinking_message,
                    content=thinking,
                    # metadata={"title": "🤔 Thinking", "status": "done"},
                ),
                ChatMessage(role="assistant", content=answer),
            ]
    yield chat_history + [
        replace(
            thinking_message,
            content=thinking,
            metadata={"title": "🤔 Thinking", "status": "done"},
        ),
        ChatMessage(role="assistant", content=answer),
    ]


with gr.Blocks() as demo:
    gr.Markdown("# Reasoning LLM Chat")

    chatbot = gr.Chatbot(
        type="messages",
        value=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Always delimit latex with $$ and $$.",
            }
        ],
        show_label=False,
        bubble_full_width=False,
        avatar_images=(None, "share/deepseek-logo-icon.svg"),
    )

    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter your message here...",
            container=False,
        )
        clear = gr.ClearButton([msg, chatbot], scale=1)

    examples = gr.Examples(
        [
            """If i have 3 fruits, an apple, two bananas and an orange on a green plate in a kitchen with a water leak from the ceiling and i climb up on a ladder carrying the plate and then flipping it upside down before using it to stop the water leak by taping it to the ceiling, how many fruits are on the plate when i get off the ladder?""",
            """Lets play a game where there is a 20 sided die on a table. It starts with a random number facing up. You can choose to collect the amount in dollars shown by the die which counts for a turn, or you can use your turn to reroll the die try to collect a higher amount on your next turn. After collecting the money, you do not have to reroll the die if you dont want to. You have 100 turns to maximize the amount of money you can make, what strategy should you use to maximize earnings, and how much can you expect to earn with this strategy? For example, if the dice starts with the number 5 facing up, you could collect $5 100 times for a total of $500, or you could try to roll to get a higher number and use your remaining turns to collect. a key aspect of the problem is that you dont have to reroll if you dont want to and can just successively keep collecting the amount from a previous roll. Lets say it takes you 10 turns to roll the number 18, then you have 90 turns left to just repeatedly collect $18. Formalize this and solve it mathematically""",
        ],
        inputs=msg,
    )

    msg.submit(
        fn=user_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    ).then(
        fn=process_llm_stream,
        # fn=process_reasoning_stream,
        inputs=chatbot,
        outputs=chatbot,
    )

demo.queue().launch()
