import json
import asyncio
import tqdm
import time

# project specific 
import importlib
llm_call = importlib.import_module("llm-call")

AZURE_OPENAI_GPT4 = "gpt4-new"
AZURE_OPENAI_GPT3p5 = "gpt3-5"
CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"

async def main():
    system_prompt = "You are a helpful assistant."
    prompt = "Can you share an example of how someone found their passion through reflecting on joyful moments? \ud83c\udf1f\ud83e\udd14"

    for _ in tqdm.tqdm(range(50)):
        response_dict = await llm_call.websocket_post_model_adapter(
            AZURE_OPENAI_GPT3p5, 
            prompt, 
            system_prompt, 
            temperature=None) 
        response_text = response_dict['result']
        print(len(response_text))
        time.sleep(5)

if __name__ == '__main__':
    asyncio.run(main())
