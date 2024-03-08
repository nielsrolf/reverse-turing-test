import json
import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


oai = OpenAI()
pplx = OpenAI(api_key=os.environ['PERPLEXITY'], base_url="https://api.perplexity.ai")


def fix_json(json_string):
    """parse partially streamed json"""
    stack = []
    inside_string = False
    escape = False

    for char in json_string:
        if char == '"' and not escape:
            inside_string = not inside_string

        if not inside_string:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif (char == '}' or char == ']') and stack:
                stack.pop()

        escape = (char == '\\' and not escape)

    # Append all unclosed elements in reverse order
    
    closing = "".join(stack[::-1])
    while len(json_string) > 0:
        try:
            return json.loads(json_string + closing)
        except json.JSONDecodeError:
            json_string = json_string[:-1]


async def aprint(data):
    print(json.dumps(data))


async def stream(
    client,
    model,
    messages,
    json_clb=aprint,
    completions_kwargs={}
):
    """Stream that parsed partial json after every new chunk and calls json_clb"""
    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **completions_kwargs
    )
    content = ""
    for chunk in response_stream:
        content += (chunk.choices[0].delta.content or "")
        print(content)
        parsed = fix_json(content)
        await json_clb(parsed)
    return parsed



    
    
async def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user. You also start every message with a variation of 'Yo yo yoo', to add some style and you always return json"
            ),
        },
        {
            "role": "user",
            "content": (
                "How many stars are in the universe?"
            ),
        },
    ]
    result = await stream(pplx, "mistral-7b-instruct", messages)
    result = await stream(oai, "gpt-3.5-turbo", messages)
    breakpoint()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
