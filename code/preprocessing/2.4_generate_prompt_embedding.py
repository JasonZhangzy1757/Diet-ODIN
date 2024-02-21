from transformers import LlamaTokenizer, LlamaModel
import torch
from tqdm import tqdm
import numpy as np

tokenizer = LlamaTokenizer.from_pretrained('../llama-2-7b')
model = LlamaModel.from_pretrained('../llama-2-7b')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

graph = torch.load('../processed_data/heterogeneous_graph_768_no_med_with_prompt_10_imbalanced.pt')
sentences = graph['food']['prompt']
batch_size = 16

model.to('cuda')
all_sentence_embeddings = []
for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
    if i + batch_size > len(sentences):
        batch_sentences = sentences[i:]
    else:
        batch_sentences = sentences[i:i + batch_size]
    inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the hidden states (last layer)
    last_hidden_state = outputs.last_hidden_state
    # Average the hidden states to get sentence embedding
    sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().detach().numpy()
    all_sentence_embeddings.append(sentence_embedding)

all_sentence_embeddings = torch.tensor(np.concatenate(all_sentence_embeddings, axis=0), dtype=torch.float32)
graph['food']['prompt_embedding'] = all_sentence_embeddings

torch.save(graph, '../processed_data/heterogeneous_graph_768_no_med_with_prompt_10_imbalanced.pt')

