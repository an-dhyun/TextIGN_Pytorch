import argparse
from model import GNN
import torch
from dataloader import TextIGNGraphDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.multiprocessing
from sklearn.metrics import f1_score, accuracy_score, recall_score

    
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Pytorch TextIGN Training')
parser.add_argument('--dataset', default='R8Fake', help='Training dataset')  # 'mr','ohsumed','R8','R52'
parser.add_argument('--test_epoch', default=50, type=int, help='Number of epochs to train.')

args = parser.parse_args()


def test(net, test_dataloader):
    print("test started!!!!")
    net.eval()

    labels = []
    prediction_list = []
    for iteration, data in enumerate(test_dataloader):
        adj, mask, emb, y = data
        adj = adj.float().cuda()
        mask = mask.float().cuda()
        emb = emb.float().cuda()
        y = y.float()
        y = torch.argmax(y, dim=1)
        labels.append(y.detach().numpy())
                
        output = net(emb, adj, mask)
        output = output.cpu()

        predicted = torch.argmax(output, dim=1)
        prediction_list.append(predicted.detach().numpy())
    
    labels = np.vstack(labels)
    prediction = np.vstack(prediction_list)
    
    accuracy = accuracy_score(labels, prediction)
    print("accuracy : ", accuracy)
    f1 = f1_score(labels, prediction)
    print("f1_score : ", f1)
    recall = recall_score(labels, prediction)
    print("recall_score : ", recall)
    


if __name__ == '__main__':
    epoch = args.test_epoch

    net = GNN(input_dim=200, hidden_dim=96, output_dim=2).cuda()
    net = net.to('cuda')
    net.load_state_dict(torch.load('./checkpoint/net_epoch_{}.pth'.format(epoch)))

    test_dataset = TextIGNGraphDataset(dataset=args.dataset, root_dir='dataloader', name='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        test(net, test_dataloader)
