import torch
import argparse
from transformers import BertTokenizer
import numpy as np


if __name__ == '__main__':

    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.current_device())

    # argparse text string 
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()

    model = torch.load('./model')
    model.eval()

    topic = ['Biology', 'Physics', 'Chemistry']

    text = args.text

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    MAX_LEN = 512
    encodings = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


    with torch.no_grad():
        input_ids = encodings['input_ids'].to('cuda', dtype=torch.long)
        attention_mask = encodings['attention_mask'].to('cuda', dtype=torch.long)
        token_type_ids = encodings['token_type_ids'].to('cuda', dtype=torch.long)
        output = model(input_ids, attention_mask, token_type_ids)
        final_output = torch.sigmoid(output[0]).cpu().detach().numpy().tolist()
        print(topic[int(np.argmax(final_output, axis=1))])