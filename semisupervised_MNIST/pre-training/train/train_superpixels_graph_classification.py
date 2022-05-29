"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import dgl
import train.aug as aug
import pdb
from train.aug import AverageMeter
import time
import copy


def train_epoch(model, GNN_imp_estimator, optimizer, device, data_loader, epoch, drop_percent, temp=0.5, aug_type='nn'):
    
    model.train()
    GNN_imp_estimator.train()
    epoch_loss = 0

    t0 = time.time()
   
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):

        torch.set_grad_enabled(False)
        node_imp = GNN_imp_estimator.forward(batch_graphs,
                                             torch.unsqueeze(batch_graphs.ndata['feat'][:, 0], 1).to(device),
                                             batch_snorm_n.to(device)).detach()
        batch_graphs.ndata['node_imp'] = node_imp

        aug_batch_graphs = dgl.unbatch(batch_graphs)
        aug_list1, aug_list2 = aug.aug_double_ex(aug_batch_graphs, aug_type)

        batch_graphs, batch_snorm_n, batch_snorm_e = aug.collate_batched_graph(aug_list1)
        aug_batch_graphs, aug_batch_snorm_n, aug_batch_snorm_e = aug.collate_batched_graph(aug_list2)

        aug_batch_x = aug_batch_graphs.ndata['feat'].to(device)
        aug_batch_snorm_n = aug_batch_snorm_n.to(device)

        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_snorm_n = batch_snorm_n.to(device)

        torch.set_grad_enabled(True)
        optimizer.zero_grad()

        ori_node_imp = GNN_imp_estimator.forward(batch_graphs, torch.unsqueeze(batch_x[:, 0], 1), batch_snorm_n)
        ori_vector = model.forward_imp(batch_graphs, batch_x, batch_snorm_n, ori_node_imp)

        aug_node_imp = GNN_imp_estimator.forward(aug_batch_graphs, torch.unsqueeze(aug_batch_x[:, 0], 1), aug_batch_snorm_n)
        aug_vector = model.forward_imp(aug_batch_graphs, aug_batch_x, aug_batch_snorm_n, aug_node_imp)

        x1_abs = ori_vector.norm(dim=1)
        x2_abs = aug_vector.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', ori_vector, aug_vector) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temp)

        pos_sim = sim_matrix[range(len(aug_list1)), range(len(aug_list1))]

        ra_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        ra_loss = - torch.log(ra_loss).mean()

        contrastive_loss = ra_loss
        # sim_matrix_tmp2 = aug.sim_matrix2(ori_vector, aug_vector, temp=temp)
        # row_softmax = nn.LogSoftmax(dim=1)
        # row_softmax_matrix = -row_softmax(sim_matrix_tmp2)
        #
        # colomn_softmax = nn.LogSoftmax(dim=0)
        # colomn_softmax_matrix = -colomn_softmax(sim_matrix_tmp2)
        #
        # row_diag_sum = aug.compute_diag_sum(row_softmax_matrix)
        # colomn_diag_sum = aug.compute_diag_sum(colomn_softmax_matrix)
        # contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))
        
        contrastive_loss.backward()
        optimizer.step()
        epoch_loss += contrastive_loss.detach().item()

        if iter % 20 == 0:
            if iter == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Aug: [{}]  Epoch: [{}/{}]  Iter: [{}/{}]  Loss: [{:.4f}]  Time Taken: [{:.2f} min]'
                 .format(aug_type, epoch, 100, iter, len(data_loader), contrastive_loss, t1 / 60))
            
    epoch_loss /= (iter + 1)
    print('Epoch: [{:>2d}]  Loss: [{:.4f}]'.format(epoch + 1, epoch_loss))

    return epoch_loss, optimizer
