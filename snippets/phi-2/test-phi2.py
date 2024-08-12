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

proj_dir = pathlib.Path().resolve()
data_dir = os.path.join(proj_dir, "data") 
config_path = os.path.join(proj_dir, "snippets", "phi-2", "config.json")

def read_config():
    with open(config_path) as f:
        return json.load(f)

def load_models(glb_config): 
    torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        glb_config["model_list"][glb_config["model_name"]], 
        torch_dtype = "auto", 
        trust_remote_code = True)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        glb_config["model_list"][glb_config["model_name"]], 
        trust_remote_code = True)
    return (model, tokenizer) 

def is_standard_english(text):
    # This regex pattern matches standard English characters, numbers, and basic punctuation
    pattern = r'^[a-zA-Z0-9\s.,!?()-]+$'
    return bool(re.match(pattern, str(text))) 

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

        if glb_config["substitute_emoji"]: 
            # Substitute emoji characters with their names
            for df_index, df in enumerate(dataframes):
                dataframes[df_index]['prompt'] = dataframes[df_index]['prompt'].apply(
                    lambda x: emoji.demojize(x))
                dataframes[df_index]['response'] = dataframes[df_index]['response'].apply(
                    lambda x: emoji.demojize(x))
                
        if glb_config["eliminate_outliers"]: 
            for df_index, df in enumerate(dataframes):
                # eliminate outliers
                dataframes[df_index] = df[df['time_taken (s)'] < 1000] 
                
        if glb_config["std_english_only"]: 
            for df_index, df in enumerate(dataframes):
                # Delete non-standard characters 
                dataframes[df_index] = df.applymap(lambda x: x if is_standard_english(x) else None)
                dataframes[df_index] = df.dropna() 
                
        if glb_config["overwrite_after_preprocess"]: 
            # Save dataframe to csv
            df_gpt3.to_csv(glb_config["path_to_dataset"], index=False) 
            
        return dataframes[0]
    
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
                dataframes[df_index]['prompt'] = dataframes[df_index]['prompt'].apply(
                    lambda x: emoji.demojize(x))
                dataframes[df_index]['response'] = dataframes[df_index]['response'].apply(
                    lambda x: emoji.demojize(x))
                
        if glb_config["eliminate_outliers"]: 
            for df_index, df in enumerate(dataframes):
                # eliminate outliers
                dataframes[df_index] = df[df['time_taken (s)'] < 1000] 
                
        if glb_config["std_english_only"]: 
            for df_index, df in enumerate(dataframes):
                # Delete non-standard characters 
                dataframes[df_index] = df.applymap(lambda x: x if is_standard_english(x) else None)
                dataframes[df_index] = df.dropna() 
                
        if glb_config["overwrite_after_preprocess"]: 
            # Save dataframe to csv
            df_gpt3.to_csv(glb_config["path_to_dataset"], index=False) 
            
        return dataframes[0]

# Randomly select samples
def sample(dataframe, glb_config): 
    # numpy random generator 
    random_generator = np.random.default_rng()
    
    examples = []
    example_num = glb_config["sample_count"]

    sample_policy = glb_config["sample_policy"]

    if sample_policy == "random":
        # df_gpt3 = df_gpt3.applymap(lambda x: x if is_long_enough(x, 5) else None)
        # df_gpt3 = df_gpt3.dropna()
        df_gpt3_sample = dataframe.sample(
            n = example_num + 1, 
            random_state = random_generator)
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

def generate(model, tokenizer, examples, example_input, glb_config): 
    result_df = pd.DataFrame(columns = ['prompt', 'response', 'ground_truth', 'resp_len', 'gt_len', 'diff'])

    sys_prompt = """Given a prompt, respond in the same length as GPT-3.5 does. You should return with one single response only. \n\n\n"""
    example_prompt = """Here are some examples. Each example consists of a prompt and a response. \n\n\n"""

    for i in tqdm.tqdm(range(glb_config["test_count"])): 
        for example in examples:
            example_prompt += example
        input = """Now, the prompt for you to predict is: """ + example_input['prompt'] + "\n\n\n"
        prompt = sys_prompt + example_prompt + input
        print("Prompt: \n" + prompt)
        inputs = tokenizer(
            prompt, 
            return_tensors = "pt", 
            return_attention_mask = False)

        max_new_tokens = glb_config["max_new_tokens"]
        outputs = model.generate(
            **inputs, 
            max_new_tokens = max_new_tokens)
        text = tokenizer.batch_decode(outputs)[0]
        
        print("Prediction: \n" + text) 
        print ("\n\n\n --------------- \n\n\n")
        print("Ground truth: \n" + example_input['response'])
        
        # append to the result dataframe
        new_record = pd.DataFrame.from_records([{
            'prompt':example_input['prompt'], 
            'response': text, 
            'ground_truth': example_input['response'], 
            'resp_len': len(text), 
            'gt_len': len(example_input['response']), 
            'diff': len(text) - len(example_input['response'])
        }])
        result_df = pd.concat(
            [result_df, new_record])
    
    # save result   
    result_df.to_csv(
        "../../data/result.csv", 
        index=False) 
    
def main():
    glb_config = read_config() 
    dataframe = prepare_datasets(glb_config)
    (examples, example_prompt) = sample(dataframe, glb_config)
    with torch.no_grad(): 
        model, tokenizer = load_models(glb_config)
        generate(model, tokenizer, examples, example_prompt, glb_config)

if __name__ == "__main__":
    main()