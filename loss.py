from torch import nn

class XiaoqLoss(nn.Module):
    def __init__(self):
        super.__init__()
        self.location_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()

    # predict 尺寸[n,8]
    def forward(self, predicts, targets):
        predict_locations = predicts[:, 0:4]
        predict_class_id_locations = predicts[:, 4:8]
        target_locations = targets[:, 0:4]
        target_class_id_locations = targets[:, 4:8]

        location_loss_value = self.location_loss(predict_locations, target_locations)
        classification_loss_value = self.classification_loss(predict_class_id_locations, target_class_id_locations)

        return location_loss_value + classification_loss_value