import openai
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from scipy.spatial.distance import cosine

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

class TextualEmbeddingModel:
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        """
        Initializes the TextualEmbeddingModel with a pre-trained model.
        
        Parameters:
        model_name (str): Name of the pre-trained model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, text):
        """
        Generates an embedding for the given text.
        
        Parameters:
        text (str): The text to generate an embedding for.
        
        Returns:
        np.array: The generated embedding.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def chatgpt_prompt(prompt):
    """
    Sends a prompt to the ChatGPT API and gets the response.
    
    Parameters:
    prompt (str): The prompt to send to ChatGPT.
    
    Returns:
    list: A list of generated labels.
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=None
    )
    result = response.choices[0].text.strip()
    return [label.strip() for label in result.split(',')]

def deduplicate_labels(enriched_labels, model, threshold):
    """
    Deduplicates labels based on semantic meaning using cosine similarity.
    
    Parameters:
    enriched_labels (list): List of enriched labels.
    model (TextualEmbeddingModel): Textual embedding model.
    threshold (float): Deduplication threshold.
    
    Returns:
    list: List of deduplicated labels.
    """
    embeddings = [model.get_embedding(label) for label in enriched_labels]
    unique_labels = []
    unique_embeddings = []

    for i, emb in enumerate(embeddings):
        is_unique = True
        for u_emb in unique_embeddings:
            if cosine(emb, u_emb) < threshold:
                is_unique = False
                break
        if is_unique:
            unique_labels.append(enriched_labels[i])
            unique_embeddings.append(emb)

    return unique_labels

def main(dataset, labels, model, threshold):
    """
    Main function to generate enriched labels and deduplicate them.
    
    Parameters:
    dataset (str): Name of the dataset.
    labels (list): List of labels.
    model (TextualEmbeddingModel): Textual embedding model.
    threshold (float): Deduplication threshold.
    
    Returns:
    list: List of deduplicated labels.
    """
    enriched_labels = set()
    for label in labels:
        prompt = f"You are a domain expert in the {dataset} dataset, helping develop a labeling system. Generate additional labels that will enrich {label} label of {dataset} dataset. Format output as comma separated labels."
        result = chatgpt_prompt(prompt)
        enriched_labels.update(result)
    
    deduplicated_labels = deduplicate_labels(list(enriched_labels), model, threshold)
    return deduplicated_labels
