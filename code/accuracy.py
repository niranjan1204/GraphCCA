import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, argparse
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# define the model
class Predictor(nn.Module):
    def __init__(self, input_size):
	super(Predictor,self).__init__()
    	self.l1 = nn.Linear(input_size, input_size/2)
        self.drop = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.sig(torch.dot(self.tanh(self.l1(self.drop(x[0]))), self.tanh(self.l1(self.drop(x[1])))))


# define the score function
def score(edges_pos, edges_neg, Z, model):
    labels_all = []
    preds_all = []
    
    for e in edges_pos:
        z1 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[0]]),0), requires_grad=False)
        z2 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[1]]),0), requires_grad=False)
        p = model(torch.cat((z1, z2))).item()

	preds_all.append(p)
        labels_all.append(1)
    
    for e in edges_neg:
	z1 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[0]]),0), requires_grad=False)
        z2 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[1]]),0), requires_grad=False)
        p = model(torch.cat((z1, z2))).item()

        preds_all.append(p)
        labels_all.append(0)
	
    roc_score = roc_auc_score(labels_all, preds_all)
    return roc_score

def trainmodel(Z, embed_size):
    model = Predictor(embed_size)
    for param in model.parameters():
        param.requires_grad=True
    
    train_edges = np.load(train_edges_path)
    train_edges_false = np.load(train_edges_false_path)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    epoch = 0
    count = 0
    loss = 0

    while(1):
        epoch += 1
        lossold = loss
        loss = 0
        for e in train_edges:
            z1 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[0]]),0), requires_grad=False)
            z2 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[1]]),0), requires_grad=False)

            p = model(torch.cat((z1, z2)))
            loss +=  -torch.log10(p+1e-4)

        for e in train_edges_false:
            z1 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[0]]),0), requires_grad=False)
            z2 = torch.autograd.Variable(torch.unsqueeze(torch.Tensor(Z[e[1]]),0), requires_grad=False)

            p = model(torch.cat((z1, z2)))
            loss +=  -torch.log10(1+1e-4 - p)

        optimizer.lr = 1/(5*epoch + 25)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(abs(lossold-loss) < 0.01*loss):
            count += 1
        else:
            count = 0
        if((count == 10 and epoch > 200) or (epoch == 400)):
            break

    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)
    
    global train_edges_path, train_edges_false_path, test_edges_path, test_edges_false_path, results_path

    train_edges_path = config["train_edges"]
    train_edges_false_path = config["train_edges_false"]
    test_edges_path = config["test_edges"]
    test_edges_false_path = config["test_edges_false"]
    Z_matrix = config["z_matrix"]
    A_matrix = config["a_matrix"]
    C_matrix = config["c_matrix"]
    results_path = config["results"]

    Z = np.load(Z_matrix)
    A = np.load(A_matrix)
    C = np.load(C_matrix)
    
    N = Z.shape[0]
    Z = Z*N/np.linalg.norm(Z)
    A = A*N/np.linalg.norm(A)
    C = C*N/np.linalg.norm(C)

    model1 = trainmodel(Z, 40)
    model2 = trainmodel(A, 50)
    model3 = trainmodel(C, 100)

    test_edges = np.load(test_edges_path)
    test_edges_false = np.load(test_edges_false_path)

    roc_score_Z = score(test_edges, test_edges_false, Z, model1)
    roc_score_A = score(test_edges, test_edges_false, A, model2)
    roc_score_C = score(test_edges, test_edges_false, C, model3)
    
    f1 = open(results_path, 'w+')
    f1.write('Test ROC score for Z: ' + str(roc_score_Z) + '\n')
    f1.write('Test ROC score for A: ' + str(roc_score_A) + '\n')
    f1.write('Test ROC score for C: ' + str(roc_score_C) + '\n')
    f1.close()

if __name__ == '__main__':
    main()
