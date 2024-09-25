import pandas as pd 
import pathlib
import os
import asyncio
import emoji
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# project specific 
import filters

proj_dir = pathlib.Path().resolve()
data_dir = os.path.join(proj_dir, "data") 

def load_models(model_name, glb_config): 
    torch.set_default_device("cpu")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, 
    #     torch_dtype = "auto", 
    #     trust_remote_code = True)
    # model = model.eval()
    model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code = True)
    return (model, tokenizer) 

async def main():
    model_gpt3, tokenizer_gpt3 = load_models("Xenova/gpt-3.5-turbo", None)
    model_gpt4, tokenizer_gpt4 = load_models("Xenova/gpt-4o", None) 
    models = [model_gpt3, model_gpt4]
    tokenizers = [tokenizer_gpt3, tokenizer_gpt4]
    
    df_raw = pd.read_pickle(os.path.join(data_dir, "latest_dataset.pkl"))
    df_raw.loc[:, 'message'] = df_raw['message'].apply(
        lambda x: emoji.demojize(x))
    df_raw.loc[:, 'response'] = df_raw['response'].apply(
        lambda x: emoji.demojize(x))
    
    df_gpt3 = df_raw[df_raw['model'].str.contains('gpt3-5')]
    df_gpt4 = df_raw[df_raw['model'].str.contains('gpt-4o')]
    
    dataframes = [df_gpt3, df_gpt4]
    
    for df_index, df in enumerate(dataframes): 
        # Delete non-standard characters 
        df.loc[:, 'message'] = df['message'].apply(
            lambda x: x if filters.is_standard_english(x) else None)
        df.loc[:, 'response'] = df['response'].apply(
            lambda x: x if filters.is_standard_english(x) else None)
        df = df.dropna(subset=['message', 'response'])
        dataframes[df_index] = df
    
    for df_index, df in enumerate(dataframes): 
        df = df.copy()
        df.loc[:, 'message_tokens'] = None
        df.loc[:, 'response_tokens'] = None
        df.loc[:, 'latency'] = None
        dataframes[df_index] = df
        # for each row, tokenize the message and response columns
        for row_index, row in df.iterrows():
            message = row['message']
            response = row['response']
            start_time = float(row['start_timestamp'])
            end_time = float(row['end_timestamp'])
            latency = end_time - start_time
            message_tokens = tokenizers[df_index].tokenize(message)
            response_tokens = tokenizers[df_index].tokenize(response)
            df.at[row_index, 'message_tokens'] = len(message_tokens)
            df.at[row_index, 'response_tokens'] = len(response_tokens)
            df.at[row_index, 'latency'] = latency
            
    for df_index, df in enumerate(dataframes):
        # eliminate outliers
        dataframes[df_index] = df[df['latency'] < 10]
            
    dataframes[0].to_csv(os.path.join(data_dir, "latest_dataset_gpt3.csv"), index=False, encoding='utf-8')  # index=False to exclude row numbers 
    dataframes[1].to_csv(os.path.join(data_dir, "latest_dataset_gpt4.csv"), index=False, encoding='utf-8')  # index=False to exclude row numbers 
    
    sns.set_theme(style="whitegrid")
    # for each data frame
    for df_index, df in enumerate(dataframes): 
        model_name = df['model'].iloc[0]
        
        # input scatter
        plt.clf()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='message_tokens', y='latency', hue='model')
        plt.xlabel('Prompt Tokens')
        plt.ylabel('Latency')
        plt.title(model_name + '_input')
        plt.savefig('data/scatter_' + model_name + '_input' + '.png') 
        
        # output scatter 
        plt.clf()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='response_tokens', y='latency', hue='model')
        plt.xlabel('Response Tokens')
        plt.ylabel('Latency')
        plt.title(model_name + '_output')
        plt.savefig('data/scatter_' + model_name + '_output' + '.png')
        
        # input cdf 
        plt.clf() 
        sns.ecdfplot(data=df['message_tokens'])
        plt.xlabel('message_tokens')
        plt.ylabel('Cumulative Probability')
        plt.title(model_name +'_CDF' +  '_input')
        plt.savefig('data/cdf_' + model_name + '_input' + '.png')
        
        # output cdf 
        plt.clf() 
        sns.ecdfplot(data=df['response_tokens'])
        plt.xlabel('response_tokens')
        plt.ylabel('Cumulative Probability')
        plt.title(model_name +'_CDF' +  '_output')
        plt.savefig('data/cdf_' + model_name + '_output' + '.png')
        
        # latency cdf 
        plt.clf() 
        sns.ecdfplot(data=df['latency'])
        plt.xlabel('latency')
        plt.ylabel('Cumulative Probability')
        plt.title(model_name +'_CDF' +  '_latency')
        plt.savefig('data/cdf_' + model_name + '_latency' + '.png')

if __name__ == "__main__":
    asyncio.run(main())
    