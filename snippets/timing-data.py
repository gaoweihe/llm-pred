import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

# tokenizers 
from transformers import GPT2TokenizerFast
claude_tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')
gpt_4o_tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4o')
gpt_3_tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo')

# Load the CSV file into a DataFrame
timing_data = pd.read_csv('data/timing_data.csv')

# filter by model name 
df_gpt3 = timing_data[timing_data['model'].str.contains('gpt3')]
df_gpt4 = timing_data[timing_data['model'].str.contains('gpt-4')]
df_gpt4_new = timing_data[timing_data['model'].str.contains('gpt4-new')]
df_claude_opus = timing_data[timing_data['model'].str.contains('claude-3-opus')]
df_claude_sonnet = timing_data[timing_data['model'].str.contains('claude-3-sonnet')] 
df_claude_haiku = timing_data[timing_data['model'].str.contains('claude-3-haiku')] 

# Display the first few rows of the DataFrame
# print(df_gpt3.head())
# print(df_gpt4.head())
# print(df_claude_sonnet.head())
# print(df_claude_haiku.head())

# df_gpt3 = df_gpt3[df_gpt3['time_taken (s)'] < 1000]
dataframes = [df_gpt3, df_gpt4, df_gpt4_new, df_claude_opus, df_claude_sonnet, df_claude_haiku]
tokenizers = [gpt_3_tokenizer, gpt_4o_tokenizer, gpt_4o_tokenizer, claude_tokenizer, claude_tokenizer, claude_tokenizer]

for df_index, df in enumerate(dataframes):
    # eliminate outliers
    dataframes[df_index] = df[df['time_taken (s)'] < 1000]

# for each data frame
for df_index, df in enumerate(dataframes):
    
    tokenizer = tokenizers[df_index]
    prompt_token_counts = []
    response_token_counts = []

    # for each row, tokenize the prompt and response columns 
    for row_index, row in df.iterrows():
        prompt = row['prompt']
        response = row['response']

        # tokenize the prompt
        prompt_tokens = tokenizer.encode(prompt)
        prompt_token_counts.append(len(prompt_tokens))

        # tokenize the response
        response_tokens = tokenizer.encode(response)
        response_token_counts.append(len(response_tokens))

        # print the tokens
        # print(prompt_tokens, response_tokens)
    
    # add the token counts to the data frame
    df['prompt_token_count'] = prompt_token_counts
    df['response_token_count'] = response_token_counts

    # add the sum of tokens to the data frame
    df['total_tokens'] = df['prompt_token_count'] + df['response_token_count']

# Save the updated data frames to new CSV files
df_gpt3.to_csv('data/timing_data_gpt3.csv', index=False)
df_gpt4.to_csv('data/timing_data_gpt4.csv', index=False)
df_gpt4_new.to_csv('data/timing_data_gpt4_new.csv', index=False)
df_claude_sonnet.to_csv('data/timing_data_claude_sonnet.csv', index=False)
df_claude_haiku.to_csv('data/timing_data_claude_haiku.csv', index=False)
df_claude_opus.to_csv('data/timing_data_claude_opus.csv', index=False)

sns.set_theme(style="whitegrid")
# scatter plot
for df in dataframes: 
    model_name = df['model'].iloc[0]
    # Plot each dataframe with observations of time_taken vs total_tokens 
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='total_tokens', y='time_taken (s)', hue='model')
    plt.xlabel('Total Tokens (Prompt + Response)')
    plt.ylabel('Time Taken (seconds)')
    plt.title(model_name)
    plt.savefig('data/scatter_' + model_name + '.png')
    # plt.show()

# violin plot 
for df in dataframes: 
    min_x_value = 0
    max_x_value = df['total_tokens'].max()
    bins = range(min_x_value, max_x_value + 50, 50)  # Bins every 50 units
    labels = [f'{i}-{i+49}' for i in range(min_x_value, max_x_value, 50)]  # Corresponding labels
    df['binned_total_tokens'] = pd.cut(df['total_tokens'], bins=bins, labels=labels, include_lowest=True)
    model_name = df['model'].iloc[0]
    plt.clf()
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='binned_total_tokens', y='time_taken (s)', inner='box')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.xlabel('Bucketed Total Tokens (Prompt + Response)')
    plt.ylabel('Time Taken (seconds)')
    plt.title(model_name)
    plt.savefig('data/violin_' + model_name + '.png')
    # plt.show()

