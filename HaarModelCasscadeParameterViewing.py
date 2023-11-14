import torch
import torch.nn as nn


# 模型定义
class WeakClassifier(nn.Module):
    def __init__(self, n_features):
        super(WeakClassifier, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = self.sigmoid(y_pred)
        y_pred = torch.round(y_pred).squeeze(-1)
        return y_pred

    def fit(self, x, y, sample_weight):
        criterion = nn.BCELoss(reduction='none')
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        for _ in range(100):
            y_pred = self(x)
            y = y.float()
            loss = criterion(y_pred, y)
            loss = (loss * sample_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def linear_parameters(self):
        return {'weight': self.linear.weight, 'bias': self.linear.bias}


class CascadeAdaBoost:
    def __init__(self, n_estimators=50, stage_threshold=0.1):
        self.n_estimators = n_estimators
        self.stage_threshold = stage_threshold
        self.estimators = []
        self.alphas = []

    def fit(self, x, y):
        n_samples, n_features = x.shape
        sample_weight = torch.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            weak_classifier = WeakClassifier(n_features)
            weak_classifier.fit(x, y, sample_weight)

            predictions = weak_classifier(x)
            misclassified = (predictions != y).float()

            error = torch.dot(sample_weight, misclassified) / sample_weight.sum()

            alpha = 0.5 * torch.log((1 - error) / error)
            self.alphas.append(alpha.item())
            self.estimators.append(weak_classifier)

            if error < self.stage_threshold:
                break

            sample_weight *= torch.exp(alpha * misclassified)
            sample_weight /= sample_weight.sum()

    def predict(self, x):
        final_predictions = torch.zeros(len(x))
        for alpha, estimator in zip(self.alphas, self.estimators):
            final_predictions += torch.tensor(alpha * estimator(x)).long()
        final_predictions = torch.sign(final_predictions)
        return final_predictions


cascade_adaboost = CascadeAdaBoost(n_estimators=5, stage_threshold=0.1)
state_dict = torch.load(r'FaceRecognition\ModelParameter\cascade_adaboost_model.pth')

# 输出弱分类器内部的参数
for estimator_state in state_dict['estimators']:
    weak_classifier = WeakClassifier(n_features=estimator_state['linear.weight'].size(1))
    weak_classifier.load_state_dict(estimator_state)
    linear_params = weak_classifier.linear_parameters()
    print("Linear Layer Weight:", linear_params['weight'])
    print("Linear Layer Bias:", linear_params['bias'])
    print()
