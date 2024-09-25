import json 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import re
import emoji
import tqdm
import pathlib
import os
import unicodedata
import time
import asyncio

# project specific 
import importlib
llm_call = importlib.import_module("llm-call")

proj_dir = pathlib.Path().resolve()
data_dir = os.path.join(proj_dir, "data") 
config_path = os.path.join(proj_dir, "snippets", "prompting.json")
np.random.seed()

def read_config():
    with open(config_path) as f:
        return json.load(f)

def load_models(model_name, glb_config): 
    torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype = "auto", 
        trust_remote_code = True)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code = True)
    return (model, tokenizer) 

def is_standard_english(text):
    # This regex pattern matches standard English characters, numbers, and basic punctuation
    pattern = r'^[a-zA-Z0-9\s.,!?()-]+$'
    return bool(re.match(pattern, str(text))) 

def is_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def is_long_enough(text, length): 
    return len(str(text)) >= length

def unicode_to_ascii(text):
    # Normalize the unicode string to decompose characters
    # The 'NFKD' normalization will replace characters with their closest ASCII equivalents
    normalized = unicodedata.normalize('NFKD', text)
    # Encode to ASCII, ignoring characters that cannot be converted
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text
    
def prepare_datasets(glb_config): 
    dataset_path = os.path.join(data_dir, glb_config["dataset"]) 
    
    if glb_config["dataset_format"] == "csv":
        # Load the CSV file into a DataFrame
        timing_data = pd.read_csv(dataset_path)

        # filter by model name 
        df_gpt3 = timing_data[timing_data['model'].str.contains('gpt3')]
        df_gpt4 = timing_data[timing_data['model'].str.contains('gpt-4')]
        df_gpt4_new = timing_data[timing_data['model'].str.contains('gpt4-new')]
        df_claude_opus = timing_data[timing_data['model'].str.contains('claude-3-opus')]
        df_claude_sonnet = timing_data[timing_data['model'].str.contains('claude-3-sonnet')] 
        df_claude_haiku = timing_data[timing_data['model'].str.contains('claude-3-haiku')] 

        dataframes = [df_gpt3, df_gpt4, df_gpt4_new, df_claude_opus, df_claude_sonnet, df_claude_haiku]
    
    elif glb_config["dataset_format"] == "json":
        dataframe_all = pd.DataFrame(columns = ['user_id', 'prompt', 'response', 'model'])
        
        # Load the JSON file into a DataFrame
        data_original = json.load(open(dataset_path))
        user_list = list(data_original.items())
        
        for user in user_list: 
            user_id = user[0]
            chat_history = user[1]
        
            for i in range(0, len(chat_history), 2): 
                if i + 1 < len(chat_history): 
                    prompt = chat_history[i]['prompt']
                    # prompt = unicode_to_ascii(prompt)
                    response = chat_history[i + 1]['response']
                    # response = unicode_to_ascii(response)
                    model = chat_history[i + 1]['model']
            new_record = pd.DataFrame.from_records([{
                'user_id': user_id, 
                'prompt': prompt, 
                'response': response, 
                'model': model
            }])
            dataframe_all = pd.concat(
                [dataframe_all, new_record])

        # filter by model name 
        df_gpt3 = dataframe_all[dataframe_all['model'].str.contains('gpt3')]
        df_gpt4 = dataframe_all[dataframe_all['model'].str.contains('gpt-4')]
        df_gpt4_new = dataframe_all[dataframe_all['model'].str.contains('gpt4-new')]
        df_claude_opus = dataframe_all[dataframe_all['model'].str.contains('claude-3-opus')]
        df_claude_sonnet = dataframe_all[dataframe_all['model'].str.contains('claude-3-sonnet')] 
        df_claude_haiku = dataframe_all[dataframe_all['model'].str.contains('claude-3-haiku')] 

        dataframes = [df_gpt3, df_gpt4, df_gpt4_new, df_claude_opus, df_claude_sonnet, df_claude_haiku]

    if glb_config["substitute_emoji"]: 
        # Substitute emoji characters with their names
        for df_index, df in enumerate(dataframes):
            df['prompt'] = df['prompt'].apply(
                lambda x: emoji.demojize(x))
            df['response'] = df['response'].apply(
                lambda x: emoji.demojize(x))
            dataframes[df_index] = df
                
    if glb_config["eliminate_outliers"]: 
        for df_index, df in enumerate(dataframes):
            # eliminate outliers
            df = df[df['time_taken (s)'] < 1000] 
            dataframes[df_index] = df
                
    if glb_config["std_english_only"]: 
        for df_index, df in enumerate(dataframes):
            # Delete non-standard characters 
            df['prompt'] = df['prompt'].apply(
                lambda x: x if is_standard_english(x) else None)
            df['response'] = df['response'].apply(
                lambda x: x if is_standard_english(x) else None)
            df = df.dropna(subset=['prompt', 'response'])
            dataframes[df_index] = df
            
    if glb_config["no_arabic"]: 
        for df_index, df in enumerate(dataframes): 
            df['prompt'] = df['prompt'].apply(
                lambda x: x if not is_arabic(x) else None)
            df['response'] = df['response'].apply(
                lambda x: x if not is_arabic(x) else None)
            df = df.dropna(subset=['prompt', 'response'])
            dataframes[df_index] = df
            
    if glb_config["overwrite_after_preprocess"]: 
        # Save dataframe to csv
        df_gpt3.to_csv(glb_config["path_to_dataset"], index=False) 
            
    return dataframes[0]

# Randomly select samples
def sample(dataframe, glb_config):       
    examples = []
    example_num = glb_config["sample_count"]

    sample_policy = glb_config["sample_policy"]

    if sample_policy == "random":
        # df_gpt3 = df_gpt3.applymap(lambda x: x if is_long_enough(x, 5) else None)
        # df_gpt3 = df_gpt3.dropna()
        df_gpt3_sample = dataframe.sample(
            n = example_num + 1)
        # Get the last row of the DataFrame
        example_prompt = df_gpt3_sample.iloc[-1]
        # Get all the rows but the last one
        df_gpt3_example = df_gpt3_sample.iloc[:-1]
        # Iterate through the DataFrame with index starting from 0 
        for index, data in enumerate(df_gpt3_example.iterrows()):
            current_example = ""
            current_example += "Prompt" + str(index + 1) + ": " + data[1]['prompt'] + "\n\n\n"
            current_example += "GPT Response" + str(index + 1) + ": " + data[1]['response'] + "\n\n\n"
            examples.append(current_example)
    elif sample_policy == "head": 
        df_gpt3_sample = dataframe.head(example_num)
        # Get the last row of the DataFrame
        example_prompt = df_gpt3_sample.iloc[-1]
        # Get all the rows but the last one
        df_gpt3_example = df_gpt3_sample.iloc[:-1]
        # Iterate through the DataFrame with index starting from 0 
        for index, data in enumerate(df_gpt3_example.iterrows()):
            current_example = ""
            current_example += "Prompt" + str(index + 1) + ": " + data[1]['prompt'] + "\n\n\n"
            current_example += "GPT Response" + str(index + 1) + ": " + data[1]['response'] + "\n\n\n"
            examples.append(current_example)
            
    return examples, example_prompt 

async def generate(model, tokenizer, examples, example_input, glb_config): 
    if glb_config['target'] == 'response':
        sys_prompt = """Given a prompt, you must respond in the same length as GPT-3.5 does. You must return with one single response only. \n\n\n"""
    elif glb_config['target'] == 'length': 
        sys_prompt = """Given a prompt, you must respond with the output length prediction of GPT-3.5. You must return with one number only. \n\n\n"""
    example_prompt = """Here are some examples. Each example consists of a prompt and a response. \n\n\n"""

    for example in examples:
        example_prompt += example
    input = """Now, the prompt for you to predict is: """ + example_input['prompt'] + "\n\n\n"
    
    response_text = None 
    if glb_config["llm_source"] == "local":
        response_text = generate_local(model, tokenizer, sys_prompt + example_prompt + input, glb_config)
    elif glb_config["llm_source"] == "remote":
        response_text = await generate_remote(
            model, 
            prompt = example_prompt + input, 
            system_prompt=sys_prompt, 
            glb_config = glb_config)
    
    # append to the result dataframe
    diff = 0
    if glb_config['target'] == 'response':
        diff = len(response_text) - len(example_input['response'])
    elif glb_config['target'] == 'length': 
        # FIXME: extract numerical result for analysis 
        diff = 0
    new_record = pd.DataFrame.from_records([{
        'examples': example_prompt,
        'prompt': example_input['prompt'], 
        'response': response_text, 
        'ground_truth': example_input['response'], 
        'resp_len': len(response_text), 
        'gt_len': len(example_input['response']), 
        'diff': diff
    }]) 
    
    return new_record

def generate_local(model, tokenizer, prompt, glb_config): 
    inputs = tokenizer(
        prompt, 
        return_tensors = "pt", 
        return_attention_mask = False)

    max_new_tokens = glb_config["max_new_tokens"]
    response_encoded = model.generate(
        **inputs, 
        max_new_tokens = max_new_tokens)
    response_decoded = tokenizer.batch_decode(response_encoded)[0]
    response_text = response_decoded[len(prompt):]
    
    return response_text 

async def generate_remote(model, prompt, system_prompt, glb_config): 
    temperature = glb_config["temperature"]
    response_dict = await llm_call.websocket_post_model_adapter(
        model, 
        prompt, 
        system_prompt, 
        temperature) 
    response_text = response_dict['result']
    
    return response_text
    
async def main():
    glb_config = read_config() 
    dataframe = prepare_datasets(glb_config)
    
    result_df = pd.DataFrame(columns = [
        'examples', 
        'prompt', 
        'response', 
        'ground_truth', 
        'resp_len', 
        'gt_len', 
        'diff'])
    
    model_name = glb_config["model_map"][glb_config["model_alias"]]
    model = None
    tokenizer = None
    if glb_config["llm_source"] == "local":
        with torch.no_grad(): 
            model, tokenizer = load_models(model_name, glb_config) 
            
    for i in tqdm.tqdm(range(glb_config["test_count"])): 
        if i != 0: 
            time.sleep(glb_config["llm_cooldown"])
            
        (examples, example_prompt) = sample(dataframe, glb_config)
        
        with torch.no_grad(): 
            new_record = await generate(model_name, tokenizer, examples, example_prompt, glb_config)
        
        result_df = pd.concat(
            [result_df, new_record]) 
    
    # save result   
    result_df.to_csv(
        os.path.join(data_dir, "result.csv"), 
        index=False) 

if __name__ == "__main__":
    asyncio.run(main())
