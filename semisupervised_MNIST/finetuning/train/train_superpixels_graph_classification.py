"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import dgl
import train.aug as aug
from train.metrics import accuracy_MNIST_CIFAR as accuracy


def cl_train_epoch(model, imp_estimator, optimizer, device, data_loader, epoch):
    model.train()
    imp_estimator.train()

    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    cl_loss = 0
    con_loss = 0
    aug1_cl_loss = 0
    aug2_cl_loss = 0
    
    print('Epoch [{}]: learning rate: [{:.6f}]'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):

        torch.set_grad_enabled(False)
        node_imp = imp_estimator.forward(batch_graphs,
                                         torch.unsqueeze(batch_graphs.ndata['feat'][:, 0], 1).to(device),
                                         batch_snorm_n.to(device)).detach()
        batch_graphs.ndata['node_imp'] = node_imp
        aug_batch_graphs = dgl.unbatch(batch_graphs)

        aug_list1, aug_list2 = aug.aug_double_ex(aug_batch_graphs)
        aug1_batch_graphs, aug1_batch_snorm_n, aug1_batch_snorm_e = aug.collate_batched_graph(aug_list1)
        aug2_batch_graphs, aug2_batch_snorm_n, aug2_batch_snorm_e = aug.collate_batched_graph(aug_list2)

        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)

        aug1_batch_x = aug1_batch_graphs.ndata['feat'].to(device)
        aug1_batch_e = aug1_batch_graphs.edata['feat'].to(device)
        aug1_batch_snorm_n = aug1_batch_snorm_n.to(device)
        aug1_batch_snorm_e = aug1_batch_snorm_e.to(device)

        aug2_batch_x = aug2_batch_graphs.ndata['feat'].to(device)
        aug2_batch_e = aug2_batch_graphs.edata['feat'].to(device)
        aug2_batch_snorm_n = aug2_batch_snorm_n.to(device)
        aug2_batch_snorm_e = aug2_batch_snorm_e.to(device)
        torch.set_grad_enabled(True)
        optimizer.zero_grad()

        aug1_node_imp = imp_estimator.forward(aug1_batch_graphs, torch.unsqueeze(aug1_batch_x[:, 0], 1),
                                              aug1_batch_snorm_n)
        # aug1_vector = model.forward_imp(aug1_batch_graphs, aug1_batch_x, aug1_batch_snorm_n, aug1_node_imp)
        #
        aug2_node_imp = imp_estimator.forward(aug2_batch_graphs, torch.unsqueeze(aug2_batch_x[:, 0], 1),
                                              aug2_batch_snorm_n)
        #
        # aug2_vector = model.forward_imp(aug2_batch_graphs, aug2_batch_x, aug2_batch_snorm_n, aug2_node_imp)
        #
        # x1_abs = aug1_vector.norm(dim=1)
        # x2_abs = aug2_vector.norm(dim=1)
        #
        # sim_matrix = torch.einsum('ik,jk->ij', aug1_vector, aug2_vector) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        # sim_matrix = torch.exp(sim_matrix / 0.3)
        #
        # pos_sim = sim_matrix[range(len(aug_list1)), range(len(aug_list1))]
        #
        # con_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # con_loss = - torch.log(con_loss).mean()

        # batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e, node_imp)
        # cl_loss = model.loss(batch_scores, batch_labels)

        aug1_batch_scores = model.forward(aug1_batch_graphs, aug1_batch_x, aug1_batch_e, aug1_batch_snorm_n,
                                          aug1_batch_snorm_e, aug1_node_imp)
        aug1_cl_loss = model.loss(aug1_batch_scores, batch_labels)

        aug2_batch_scores = model.forward(aug2_batch_graphs, aug2_batch_x, aug2_batch_e, aug2_batch_snorm_n,
                                          aug2_batch_snorm_e, aug2_node_imp)
        aug2_cl_loss = model.loss(aug2_batch_scores, batch_labels)

        loss = aug1_cl_loss + aug2_cl_loss #+ 0.1*con_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(aug1_batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

        if iter % 20 == 0:
            print('Iter [{}]: cl_loss [{:.4f}]  aug1_cl_loss [{:.4f}]  aug2_cl_loss [{:.4f}] con_loss [{:.4f}]  loss [{:.4f}]   Train Acc [{:.4f}]'
                  .format(iter, cl_loss, aug1_cl_loss, aug2_cl_loss, con_loss, loss, epoch_train_acc / (nb_data)))
        
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    return epoch_loss, epoch_train_acc, optimizer

def train_epoch(model, imp_estimator, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0

    print('Epoch [{}]: learning rate: [{:.6f}]'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        optimizer.zero_grad()

        node_imp = imp_estimator.forward(batch_graphs, torch.unsqueeze(batch_x[:, 0], 1), batch_snorm_n)
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e, node_imp)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

        if iter % 20 == 0:
            print('Iter [{}]: loss [{:.4f}]   Train Acc [{:.4f}]'.format(iter, loss, epoch_train_acc / (nb_data)))

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, imp_estimator, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)

            node_imp = imp_estimator.forward(batch_graphs, torch.unsqueeze(batch_x[:, 0], 1), batch_snorm_n)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e, node_imp)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc