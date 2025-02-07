import io
import boto3
import json


class LineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:
    ```
    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...
    ```

    While usually each PayloadPart event from the event stream will contain a byte array
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}
    ```

    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character) within
    the buffer via the 'scan_lines' function. It maintains the position of the last read
    position to ensure that previous bytes are not exposed again.
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


def is_thinking_message(message: dict) -> bool:
    """Check if the message is a thinking message."""
    metadata = message.get("metadata")
    if metadata:
        if "thinking" in metadata.get("title").lower():
            return True
    return False


def invoke_endpoint(history: list[dict[str, str]], **params):
    endpoint_name = "xifin-reasoner-7b-endpoint"

    payload = {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
            if not is_thinking_message(msg)
        ],
        **params,
        "stream": True,
    }
    try:
        smr = boto3.client("sagemaker-runtime", region_name="us-west-2")
        response_stream = smr.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes="accept_eula=true",
        )
        event_stream = response_stream["Body"]
        start_json = b"{"
        for line in LineIterator(event_stream):
            if line != b"" and start_json in line:
                try:
                    data = json.loads(line[line.find(start_json) :].decode("utf-8"))
                    if "choices" in data:
                        yield data["choices"][0]["delta"]["content"]
                    elif "error" in data:
                        print(f"Error encountered: {data['error']}")
                        break  # or continue based on how you want to handle errors
                    else:
                        print("Unexpected data format:", data)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except KeyError as e:
                    print(f"Key error: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
    except Exception as E:
        print("Exception while invoking llama3 endpoint: {}".format(str(E)))
        raise Exception(E)
