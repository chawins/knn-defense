import torch

import foolbox


class DkNNFoolboxModel(foolbox.models.Model):
    """Foolbox Model object wrapper for our DkNN object"""

    def __init__(self, dknn, bounds, channel_axis, preprocessing=(0, 1)):
        super(DkNNFoolboxModel, self).__init__(bounds=bounds,
                                               channel_axis=channel_axis,
                                               preprocessing=preprocessing)
        self.dknn = dknn

    def batch_predictions(self, images):
        # images is numpy array
        y_pred = self.dknn.classify(torch.tensor(images))
        return y_pred

    def num_classes(self):
        return self.dknn.num_classes
