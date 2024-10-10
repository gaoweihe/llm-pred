# may need to install websockets
# run python3 -m pip install websockets

import json
import asyncio
import websockets

AZURE_OPENAI_GPT4 = "gpt4-new"
AZURE_OPENAI_GPT3p5 = "gpt3-5"
CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"

async def websocket_post_model_adapter(model, prompt, system, temperature):
    uri = "wss://slr5qp3v97.execute-api.us-east-1.amazonaws.com/production/"
    
    if temperature is None:
        temperature = 0.7  # Default temperature value

    async with websockets.connect(uri) as websocket:
        message = json.dumps({
            "action": "runModel",
            "model": model,
            "prompt": prompt,
            "system": system,
            "temperature": temperature
        })
        await websocket.send(message)

        # Handle WebSocket closing gracefully
        try:
            response_dict = None
            while response_dict is None or "message" not in response_dict:
                response = await websocket.recv()
                response_dict = json.loads(response)
                print(response, response_dict)  # For debugging purposes
        except websockets.exceptions.ConnectionClosedOK as e:
            print(f"Connection closed: {e}")

        return response_dict

def main():
    system = "You are a helpful assistant."
    prompt = "I love Mexico. What is the best food there?"

    asyncio.run(websocket_post_model_adapter(model=AZURE_OPENAI_GPT3p5, prompt=prompt, system=system, temperature=None))

if __name__ == '__main__':
    main()
