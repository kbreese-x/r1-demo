from dataclasses import replace
import gradio as gr
from gradio import ChatMessage
from sagemaker_utils import *


def insert_system_message(chat_history):
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Always delimit latex with $$ and $$.",
    }
    if len(chat_history) == 0:
        return [system_msg]
    elif chat_history[0]["role"] != "system":
        chat_history.insert(0, system_msg)

    return chat_history


def process_llm_stream_interface(message, chat_history):
    thinking = ""
    answer = ""
    buffer = ""
    in_think = False

    # chat_history = insert_system_message(chat_history)
    chat_history.append({"role": "user", "content": message})

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
            yield [
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
            yield [replace(thinking_message, content=thinking)]
        else:
            answer += buffer
            buffer = ""
            yield [
                replace(
                    thinking_message,
                    content=thinking,
                    # metadata={"title": "🤔 Thinking", "status": "done"},
                ),
                ChatMessage(role="assistant", content=answer),
            ]
    yield [
        replace(
            thinking_message,
            content=thinking,
            metadata={"title": "🤔 Thinking", "status": "done"},
        ),
        ChatMessage(role="assistant", content=answer),
    ]


demo = gr.ChatInterface(
    fn=process_llm_stream_interface,
    chatbot=gr.Chatbot(
        type="messages",
        value=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Always delimit latex with $$ and $$.",
            }
        ],
        height="85vh",
        resizeable=True,
        show_label=False,
        bubble_full_width=False,
        avatar_images=(None, "share/deepseek-logo-icon.svg"),
    ),
    fill_height=True,
    editable=True,
    examples=[
        """If i have 3 fruits, an apple, two bananas and an orange on a green plate in a kitchen with a water leak from the ceiling and i climb up on a ladder carrying the plate and then flipping it upside down before using it to stop the water leak by taping it to the ceiling, how many fruits are on the plate when i get off the ladder?""",
        """If i have a plate with 5 fruits and 3 vegetables and i pick it up and flip it upside down, put a cup of water on top of the plate, then vigorously shake the plate with two hands, how many fruits are in the water after flipping the plate right side up again?"""
        """Lets play a game where there is a 20 sided die on a table. It starts with a random number facing up. You can choose to collect the amount in dollars shown by the die which counts for a turn, or you can use your turn to reroll the die try to collect a higher amount on your next turn. After collecting the money, you do not have to reroll the die if you dont want to. You have 100 turns to maximize the amount of money you can make, what strategy should you use to maximize earnings, and how much can you expect to earn with this strategy? For example, if the dice starts with the number 5 facing up, you could collect $5 100 times for a total of $500, or you could try to roll to get a higher number and use your remaining turns to collect. a key aspect of the problem is that you dont have to reroll if you dont want to and can just successively keep collecting the amount from a previous roll. Lets say it takes you 10 turns to roll the number 18, then you have 90 turns left to just repeatedly collect $18. Formalize this and solve it mathematically""",
    ],
    example_labels=["Fruit on plate", "Fruit on a plate (pt. 2)", "Dice game"],
)

demo.launch()
