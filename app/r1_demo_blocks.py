from dataclasses import replace
import gradio as gr
from gradio import ChatMessage
from sagemaker_utils import *


def insert_system_message(chat_history, system_prompt: str | None = None):
    if system_prompt is None:
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Always delimit latex with $$ and $$.",
        }
    else:
        system_msg = {
            "role": "system",
            "content": system_prompt,
        }
    if len(chat_history) == 0:
        return [system_msg]
    elif chat_history[0]["role"] != "system":
        chat_history.insert(0, system_msg)
    else:
        chat_history[0] = system_msg

    return chat_history


def clear_history():
    return []


def user_message(
    message: str, history: list[ChatMessage], mode: str, system_prompt: str
):
    if mode == "Text":
        return "", insert_system_message(history, system_prompt) + [
            ChatMessage(role="user", content=message)
        ]
    else:
        return "", json.loads(message)


def process_llm_stream(chat_history, params: dict):
    thinking = ""
    answer = ""
    buffer = ""
    in_think = False

    thinking_message = ChatMessage(
        role="assistant",
        content="",
        metadata={"title": "🤔 Thinking", "status": "pending"},
    )

    for chunk in invoke_endpoint(chat_history, **params):
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
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            type="messages",
            value=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Always delimit latex with $$ and $$.",
                }
            ],
            height="75vh",
            show_label=False,
            bubble_full_width=False,
            resizeable=True,
            avatar_images=(None, "share/deepseek-logo-icon.svg"),
        )

        with gr.Row():
            msg = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter your message here...",
                container=False,
            )
            mode_selector = gr.Radio(
                ["Text", "JSON History"],
                value="Text",
                label="Mode",
                interactive=True,
                container=False,
                scale=0,
            )
            clear = gr.ClearButton([msg, chatbot], scale=0)

        # examples = gr.Examples(
        #     [
        #         """If i have 3 fruits, an apple, two bananas and an orange on a green plate in a kitchen with a water leak from the ceiling and i climb up on a ladder carrying the plate and then flipping it upside down before using it to stop the water leak by taping it to the ceiling, how many fruits are on the plate when i get off the ladder?""",
        #         """Lets play a game where there is a 20 sided die on a table. It starts with a random number facing up. You can choose to collect the amount in dollars shown by the die which counts for a turn, or you can use your turn to reroll the die try to collect a higher amount on your next turn. After collecting the money, you do not have to reroll the die if you dont want to. You have 100 turns to maximize the amount of money you can make, what strategy should you use to maximize earnings, and how much can you expect to earn with this strategy? For example, if the dice starts with the number 5 facing up, you could collect $5 100 times for a total of $500, or you could try to roll to get a higher number and use your remaining turns to collect. a key aspect of the problem is that you dont have to reroll if you dont want to and can just successively keep collecting the amount from a previous roll. Lets say it takes you 10 turns to roll the number 18, then you have 90 turns left to just repeatedly collect $18. Formalize this and solve it mathematically""",
        #     ],
        #     inputs=msg,
        # )
    with gr.Tab("Settings"):
        with gr.Accordion("View Parameters JSON", open=False):
            params_json = gr.JSON(
                value={
                    "messages": [],
                    "temperature": 1.0,
                    "max_tokens": 1024,
                    "stream": True,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "top_p": 0.95,
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "stream_options": None,
                    "tool_choice": "auto",
                },
                label="Current Parameters",
                visible=True,
            )

        with gr.Group():
            gr.Markdown("## System Configuration")
            system_message = gr.TextArea(
                label="System Message",
                placeholder="You are a helpful assistant.",
                value=open("share/prompt.txt").read(),
                # value="You are a helpful assistant. Always delimit latex with $$ and $$.",
                lines=15,
            )
            model = gr.Dropdown(
                label="Model",
                choices=[
                    "xifin-reasoner-7b-endpoint",
                    "xifin-chat-llama3-1-8b-instruct-endpoint",
                ],
                value="xifin-reasoner-7b-endpoint",
                interactive=False,
            )
        with gr.Group():
            response_format = gr.TextArea(
                label="Response Format",
                placeholder="Enter JSON schema for response format",
            )

        with gr.Group():
            gr.Markdown("## Generation Parameters")
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.3,
                    step=0.05,
                    label="Temperature",
                    info="Higher values make output more random, lower values more focused",
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.95,
                    step=0.05,
                    label="Top P",
                    info="Alternative to temperature for nucleus sampling",
                )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=1024,
                    step=1,
                    label="Max Tokens",
                    info="Maximum number of tokens to generate",
                )
                n = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Completions",
                    info="Number of chat completion choices to generate",
                )

            with gr.Row():
                frequency_penalty = gr.Slider(
                    minimum=-2,
                    maximum=2,
                    value=0,
                    step=0.1,
                    label="Frequency Penalty",
                    info="Penalize frequent tokens",
                )
                presence_penalty = gr.Slider(
                    minimum=-2,
                    maximum=2,
                    value=0,
                    step=0.1,
                    label="Presence Penalty",
                    info="Penalize tokens based on presence",
                )
        with gr.Group():
            gr.Markdown("## Tool Configuration")
            tool_choice = gr.Radio(
                choices=["auto", "none"], value="auto", label="Tool Choice"
            )
            tool_prompt = gr.TextArea(
                label="Tool Prompt",
                placeholder="Enter tool prompt here",
                info="Prompt to be appended before the tools",
            )

        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                seed = gr.Number(
                    value=None,
                    label="Seed",
                    precision=0,
                    info="Random seed for reproducibility",
                )
                logprobs = gr.Checkbox(value=False, label="Return Log Probabilities")
                stream = gr.Checkbox(value=True, label="Stream Response")

            with gr.Row():
                top_logprobs = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=0,
                    step=1,
                    label="Top Log Probabilities",
                    info="Number of most likely tokens to return",
                    visible=False,
                )

            stop_sequences = gr.TextArea(
                label="Stop Sequences",
                placeholder="Enter sequences separated by newlines",
                info="The API will stop generating when it encounters these sequences",
            )

        @gr.on(
            inputs={
                temperature,
                max_tokens,
                frequency_penalty,
                presence_penalty,
                top_p,
                seed,
                logprobs,
                stream,
                top_logprobs,
                stop_sequences,
                model,
                n,
                tool_choice,
                tool_prompt,
                response_format,
            },
            outputs=params_json,
        )
        def update_params(data):
            params = {
                "temperature": data[temperature],
                "max_tokens": int(data[max_tokens]),
                "frequency_penalty": data[frequency_penalty],
                "presence_penalty": data[presence_penalty],
                "top_p": data[top_p],
                "stream": data[stream],
                "model": data[model],
                "n": int(data[n]),
                "tool_choice": data[tool_choice],
            }

            # Optional parameters
            if data[seed] is not None:
                params["seed"] = int(data[seed])

            if data[logprobs]:
                params["logprobs"] = True
                params["top_logprobs"] = (
                    int(data[top_logprobs]) if data[top_logprobs] > 0 else None
                )

            if data[stop_sequences].strip():
                params["stop"] = [
                    s.strip() for s in data[stop_sequences].split("\n") if s.strip()
                ]

            if data[tool_prompt]:
                params["tool_prompt"] = data[tool_prompt]

            if data[response_format]:
                try:
                    params["response_format"] = {
                        "type": "json_object",
                        "value": json.loads(data[response_format]),
                    }
                except json.JSONDecodeError as e:
                    gr.Warning(f"Invalid JSON in response format:\n{e}")
                    pass

            return params

        # Show/hide top_logprobs based on logprobs checkbox
        @gr.on(inputs=logprobs, outputs=top_logprobs)
        def toggle_logprobs(show_logprobs):
            return gr.update(visible=show_logprobs)

    msg.submit(
        fn=user_message,
        inputs=[msg, chatbot, mode_selector, system_message],
        outputs=[msg, chatbot],
    ).then(
        fn=process_llm_stream,
        # fn=process_reasoning_stream,
        inputs=[chatbot, params_json],
        outputs=chatbot,
    )

demo.queue().launch()
