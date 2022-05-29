import os.path as osp

from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader

from gin import Encoder, Explainer
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
import random


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, pooling, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        if pooling == 'last':
            self.embedding_dim = hidden_dim
        else:
            self.embedding_dim = hidden_dim*num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, pooling)
        self.explainer = Explainer(dataset_num_features, hidden_dim, 3)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, node_imp):

        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch, node_imp)

        y = self.proj_head(y)

        return y

    def explain(self, x, edge_index, batch):

        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y = self.explainer(x, edge_index, batch)

        return y

    def loss_cal(self, x, x_aug, x_cp):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        x_cp_abs = x_cp.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        sim_matrix_cp = torch.einsum('ik,jk->ij', x, x_cp) / torch.einsum('i,j->ij', x_abs, x_cp_abs)
        sim_matrix_cp = torch.exp(sim_matrix_cp / T)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / sim_matrix.sum(dim=1)
        loss2 = pos_sim / (sim_matrix_cp.sum(dim=1) + pos_sim)

        loss = loss1 + 0.1*loss2

        loss = - torch.log(loss).mean()

        return loss

if __name__ == '__main__':

    args = arg_parse()

    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    imp_batch_size = 2048
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    num_workers = args.num_workers
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset = TUDataset(path, name=DS, aug=args.aug, rho=args.rho)
    dataset_eval = TUDataset(path, name=DS, aug='none')
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=2*num_workers)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = simclr(args.hidden_dim, args.num_gc_layers, args.pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('dataset: {}'.format(DS))
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('pooling: {}'.format(args.pooling))
    print('================')



    for epoch in range(1, epochs+1):

        dataset.aug = "none"
        dataloader = DataLoader(dataset, batch_size=imp_batch_size, num_workers=num_workers, shuffle=False)
        model.eval()
        torch.set_grad_enabled(False)
        for step, data in enumerate(dataloader):
            node_index_start = step * imp_batch_size
            node_index_end = min(node_index_start + imp_batch_size - 1, len(dataset) - 1)
            data, data_aug, data_cp = data
            node_num, _ = data.x.size()
            data = data.to(device)
            node_imp = model.explain(data.x, data.edge_index, data.batch).detach()
            dataset.node_score[dataset.slices['x'][node_index_start]:dataset.slices['x'][node_index_end + 1]] = \
                torch.squeeze(node_imp.half())

        loss_all = 0
        dataset.aug = args.aug
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        model.train()
        torch.set_grad_enabled(True)
        for data in dataloader:

            data, data_aug, data_cp = data
            optimizer.zero_grad()
            
            node_num, _ = data.x.size()
            data = data.to(device)
            data_imp = model.explain(data.x, data.edge_index, data.batch)
            x = model(data.x, data.edge_index, data.batch, data_imp)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

                edge_idx = data_cp.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_cp = len(idx_not_missing)
                data_cp.x = data_cp.x[idx_not_missing]

                data_cp.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_cp)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                            not edge_idx[0, n] == edge_idx[1, n]]
                data_cp.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
            data_cp = data_cp.to(device)

            data_aug_imp = model.explain(data_aug.x, data_aug.edge_index, data_aug.batch)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug_imp)

            data_cp_imp = model.explain(data_cp.x, data_cp.edge_index, data_cp.batch)
            x_cp = model(data_cp.x, data_cp.edge_index, data_cp.batch, data_cp_imp)

            loss = model.loss_cal(x, x_aug, x_cp)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)

    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug + '_' + args.log, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')
