from lib.sam import SamPredictor
from lib.internvideo import InternVideoPredictor
from mmdet.datasets.builder import PIPELINES

class ActiveLearningModel:
    def __init__(self):
        """Load InternVideo for active learning"""
        self.model = InternVideoPredictor()

    def predict(self, batch):
        """Return predictions from InternVideo"""
        return self.model.predict(batch)

class ZeroShotModel:
    def __init__(self):
        """Load SAM for zero-shot segmentation"""
        self.model = SamPredictor()

    def predict(self, batch):
        """Return zero-shot segmentation results"""
        return self.model.predict(batch)

@PIPELINES.register_module()
class ActiveLearningWithZeroShot:
    def __init__(self):
        """Initialize active learning and zero-shot models"""
        self.active_model = ActiveLearningModel()
        self.zero_shot_model = ZeroShotModel()
        self.control = 0
        self.mode = "Zero"

    def __call__(self, results):
        """Process input batch"""
        batch = results['img']
        active_results = self.active_model.predict(batch)
        
        if self.control == 0:
            zero_results = self.zero_shot_model.predict(batch)
            if self.mode == "Active":
                results['ann_info'] = active_results
            else:
                results['ann_info'] = zero_results

            # Compute mAP for active learning vs zero-shot
            anno = self.zero_shot_model.predict(batch)
            if mean_average_precision(active_results, anno) > mean_average_precision(zero_results, anno):
                self.control = 1
                self.mode = "Active"
            else:
                self.control = 0
                self.mode = "Zero"
        else:
            self.control -= 1
            results['ann_info'] = active_results

        return results
