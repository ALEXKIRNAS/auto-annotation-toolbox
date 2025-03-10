import torch
import numpy as np
from lib.sam import SamPredictor
from lib.internvideo import InternVideoPredictor

class ActiveLearningModel:
    def __init__(self):
        """
        Initializes the ActiveLearningModel using the InternVideo model.
        """
        self.model = InternVideoPredictor()  # Initialize the InternVideo model

    def predict(self, batch):
        """
        Predicts using the active learning model.
        
        Parameters:
        batch (list): The batch of data.
        
        Returns:
        list: The predictions.
        """
        # Placeholder for actual prediction logic
        return self.model.predict(batch)

class ZeroShotModel:
    def __init__(self):
        """
        Initializes the ZeroShotModel using the SAM model.
        """
        self.model = SamPredictor()  # Initialize the SAM model

    def predict(self, batch):
        """
        Predicts using the zero-shot learning model.
        
        Parameters:
        batch (list): The batch of data.
        
        Returns:
        list: The predictions.
        """
        # Placeholder for actual prediction logic
        return self.model.predict(batch)

def process_dataset(dataset, labels, active_model, zero_shot_model, map_function):
    """
    Processes the dataset using active learning and zero-shot learning models.
    
    Parameters:
    dataset (list): The dataset.
    labels (list): The dataset labels.
    active_model (ActiveLearningModel): The active learning model.
    zero_shot_model (ZeroShotModel): The zero-shot learning model.
    map_function (function): The quality function (mAP).
    
    Yields:
    list: The predictions.
    """
    a, b, control = 1, 1, 0
    mode = "Zero"

    for batch in dataset:
        results = active_model.predict(batch)
        if control == 0:
            results_zero = zero_shot_model.predict(batch)
            if mode == "Active":
                yield results
            else:
                yield results_zero
            anno = zero_shot_model.predict(batch)
            if map_function(results, anno) > map_function(results_zero, anno):
                control = b
                a, b = b, a + b
                mode = "Active"
            else:
                a, b, control = 1, 1, 0
                mode = "Zero"
        else:
            control -= 1
            yield results
