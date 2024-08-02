import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import myutils as utils

class autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75), int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size*0.1),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75),int(feature_size)),
                                     )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.SmoothL1Loss()
Params = utils.get_params('AE')
EvalParams = utils.get_params('Eval')

def huber_loss_per_sample(output, target, delta=1.0):
    abs_error = torch.abs(output - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear

def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])

def train(X_train, feature_size, epoches=Params['epoches'], lr=Params['lr']):
    model = autoencoder(feature_size).to(device)
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=Params['weight_decay'])
    model.train()

    X_train = torch.from_numpy(X_train).type(torch.float)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
    torch_dataset = Data.TensorDataset(X_train, X_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Params['batch_size'],
        shuffle=True,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        if EvalParams['verbose_info']:
            print('epoch:{}/{}'.format(epoch, epoches), '|Loss:', loss.item())

    model.eval()
    output = model(X_train)
    mse_vec = huber_loss_per_sample(output, X_train).mean(dim=1).cpu().data.numpy()

    if EvalParams['verbose_info']:
        print("max AD score", max(mse_vec))

    thres = max(mse_vec)
    mse_vec.sort()
    pctg = Params['percentage']
    thres = mse_vec[int(len(mse_vec) * pctg)]

    if EvalParams['verbose_info']:
        print("thres:", thres)

    return model, thres

@torch.no_grad()
def test(model, thres, X_test):
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_test = X_test.to(device)
    output = model(X_test)
    mse_vec = huber_loss_per_sample(output, X_test).mean(dim=1).cpu().data.numpy()
    y_pred = np.asarray([0] * len(mse_vec))
    idx_mal = np.where(mse_vec > thres)
    y_pred[idx_mal] = 1

    return y_pred, mse_vec

def test_plot(mse_vec, thres, file_name=None, label=None):
    plt.figure()
    plt.plot(np.linspace(0, len(mse_vec) - 1, len(mse_vec)), [thres] * len(mse_vec), c='black', label='99th-threshold')
    # plt.ylim(0,thres*2.)

    if label is not None:
        idx = np.where(label == 0)[0]
        plt.scatter(idx, mse_vec[idx], s=8, color='blue', alpha=0.4, label='Normal')

        idx = np.where(label == 1)[0]
        plt.scatter(idx, mse_vec[idx], s=8, color='red', alpha=0.7, label='Anomalies')
    else:
        plt.scatter(np.linspace(0, len(mse_vec) - 1, len(mse_vec)), mse_vec, s=8, alpha=0.4, label='Test samples')

    plt.legend()
    plt.xlabel('Sample NO.')
    plt.ylabel('Anomaly Score (RMSE)')
    plt.title('Per-sample Score')
    if file_name is None:
        plt.show()
    else:
        plt.rcParams.update({'figure.dpi': 300})
        plt.savefig(file_name)
