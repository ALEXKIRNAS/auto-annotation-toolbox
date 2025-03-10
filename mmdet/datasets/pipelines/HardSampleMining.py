import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import VideoSwinModel, VideoSwinConfig
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos
        self.labels = labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx]

class HardSampleMining:
    def __init__(self, videos, labels, epochs, num_samples, device='cuda'):
        """
        Initializes the HardSampleMining class with the dataset and training parameters.
        
        Parameters:
        videos (list): List of video data points.
        labels (list): List of labels corresponding to the videos.
        epochs (int): Number of training epochs.
        num_samples (int): Number of hard samples to select each epoch.
        device (str): Device to use for training ('cuda' or 'cpu').
        """
        self.videos = np.array(videos)
        self.labels = np.array(labels)
        self.epochs = epochs
        self.num_samples = num_samples
        self.dataset = list(zip(self.videos, self.labels))
        self.device = device
        self.model = self.initialize_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def initialize_model(self):
        """
        Initializes the Video Swin Transformer Large model.

        Returns:
        nn.Module: The Video Swin Transformer model.
        """
        config = VideoSwinConfig.from_pretrained('microsoft/videomae-large')
        model = VideoSwinModel(config)
        model.to(self.device)
        return model

    def train_model(self, train_data):
        """
        Trains the model on the given dataset for one epoch.

        Parameters:
        train_data (list): List of training data points.
        """
        train_loader = DataLoader(VideoDataset(*zip(*train_data)), batch_size=4, shuffle=True)
        self.model.train()
        
        for videos, labels in train_loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(videos).logits
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"Training loss: {loss.item()}")

    def get_embeddings(self, data):
        """
        Generates embeddings for the given data points.

        Parameters:
        data (list): List of data points to generate embeddings for.

        Returns:
        np.array: Array of embeddings for the data points.
        """
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for video in data:
                video_tensor = torch.tensor(video).unsqueeze(0).to(self.device)
                embedding = self.model(video_tensor).pooler_output
                embeddings.append(embedding.cpu().numpy().flatten())

        return np.array(embeddings)

    def hard_sample_mining(self):
        """
        Performs hard sample mining and trains the model over multiple epochs.
        """
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            embeddings = self.get_embeddings(self.videos)
            k = self.determine_k(embeddings)
            kmeans = KMeans(n_clusters=k).fit(embeddings)
            distances = kmeans.transform(embeddings)

            hard_samples = self.select_hard_samples(distances)
            train_data = self.oversample(hard_samples)

            self.train_model(train_data)

    def determine_k(self, embeddings):
        """
        Determines the number of clusters K using the Elbow method.

        Parameters:
        embeddings (np.array): Array of embeddings.

        Returns:
        int: The number of clusters.
        """
        distortions = []
        K = range(1, 400)

        for k in K:
            kmeans = KMeans(n_clusters=k).fit(embeddings)
            distortions.append(sum(np.min(pairwise_distances_argmin_min(embeddings, kmeans.cluster_centers_)[1]) ** 2))

        # Finding the elbow point
        k_optimal = np.diff(distortions, 2).argmax() + 2
        return k_optimal

    def select_hard_samples(self, distances):
        """
        Selects hard samples based on distances from cluster centers.

        Parameters:
        distances (np.array): Array of distances from cluster centers.

        Returns:
        list: List of indices of selected hard samples.
        """
        closest_indices = np.argsort(distances, axis=0)[:self.num_samples].flatten()
        furthest_indices = np.argsort(distances, axis=0)[-self.num_samples:].flatten()
        return closest_indices.tolist() + furthest_indices.tolist()

    def oversample(self, hard_sample_indices):
        """
        Oversamples the dataset by replacing samples with selected hard samples.

        Parameters:
        hard_sample_indices (list): List of indices of selected hard samples.

        Returns:
        list: Oversampled training dataset.
        """
        hard_samples = [self.dataset[i] for i in hard_sample_indices]
        return self.dataset + hard_samples
