from losses import TripletLoss, OnlineTripletLoss
from networks import ClassificationNet, TripletNet
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class ResNetEmbeddingNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # 加载预训练的 resnet50
        resnet = models.resnet50(pretrained=True)
        # 去掉最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = output_dim
        # 新的全连接层用于降维
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FullNet(nn.Module):
    def __init__(self, num_classes):
        super(FullNet, self).__init__()
        self.resnet_output_dim = 512
        self.embedding_net = ResNetEmbeddingNet(self.resnet_output_dim)
        self.classifier = ClassificationNet(n_classes=num_classes, embedding_dim=self.resnet_output_dim)

    def forward(self, anc_img, pos_img, neg_img, label):
        anc_embedding = self.embedding_net(anc_img)
        pos_embedding = self.embedding_net(pos_img)
        neg_embedding = self.embedding_net(neg_img)
        anc_classification_output = self.classifier(anc_embedding)
        # pos_classification_output = self.classifier(pos_embedding)
        # neg_classification_output = self.classifier(neg_embedding)
        return anc_embedding, pos_embedding, neg_embedding, anc_classification_output
    
    def get_embedding(self, img):
        with torch.no_grad():
            embedding = self.embedding_net(img)
        return embedding



def make_model(num_classes):
    model = FullNet(num_classes)
    return model