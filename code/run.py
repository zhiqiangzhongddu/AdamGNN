import pandas as pd
import numpy as np
import math
import random
import networkx as nx
import time
import datetime
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, normalized_mutual_info_score
from utils import *
import sys

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torch_geometric as tg
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge
from torch_geometric.nn.pool.topk_pool import topk


def GC_run(model, optimizer, train_loader, val_loader, test_loader, model_config, device, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    writer = SummaryWriter(logdir='runs/{}'.format(model_config['alias']))
    writer.add_text('config', str(model_config), 0)

    loss_func = torch.nn.NLLLoss()

    test_results_accuracy = []
    valid_results_accuracy = []
    test_results_f1_micro = []
    valid_results_f1_micro = []
    test_results_f1_macro = []
    valid_results_f1_macro = []
    test_results_nmi = []
    valid_results_nmi = []
    ls_rm = []
    ls_scores = []

    for epoch_id, epoch in enumerate(range(model_config['num_epoch'])):
        start_epoch = time.time()
        if epoch_id % model_config['verbose'] == 0:
            print('Epoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss_recon = 0
        total_loss_kl = 0
        total_loss_nc = 0
        total_loss = 0
        
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            labels_tensor = data.y

            model.train()
            optimizer.zero_grad()
            out, loss_recon, loss_kl, _, _ = model.forward(data, epoch_id)
            loss_nc = loss_func(out, labels_tensor)

            if model_config['loss_mode']=='all':
                loss = loss_nc + 0.1 * loss_recon + 1 * loss_kl # 1 10 100 1000
            elif model_config['loss_mode']=='kl':
                loss = loss_nc + 1 * loss_kl
            elif model_config['loss_mode']=='recon':
                loss = loss_nc + 0.1 * loss_recon
            elif model_config['loss_mode']=='nc':
                loss = loss_nc

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss_recon += loss_recon.cpu().item()
            total_loss_kl += loss_kl.cpu().item()
            total_loss_nc += loss_nc.cpu().item()
            total_loss += loss.cpu().item()
        
        print(round(total_loss_nc, 5), round(total_loss_recon, 5), round(total_loss_kl, 5))
        print(round(total_loss, 5))

        # write epoch info
        writer.add_scalar('model/loss', total_loss, epoch_id)
        
        # evaluate epoch    
        if epoch_id % model_config['verbose'] == 0:
            model.eval()
            epoch_train_results_accuracy = []
            epoch_train_results_f1_micro = []
            epoch_train_results_f1_macro = []
            epoch_train_results_nmi = []
            epoch_valid_results_accuracy = []
            epoch_valid_results_f1_micro = []
            epoch_valid_results_f1_macro = []
            epoch_valid_results_nmi = []
            epoch_test_results_accuracy = []
            epoch_test_results_f1_micro = []
            epoch_test_results_f1_macro = []
            epoch_test_results_nmi = []
            # epoch_ls_rm = []
            # epoch_ls_scores = []
            

            for idx, data in enumerate(train_loader):
                labels = data.y.data.numpy()
                data = data.to(device) 
                out, _, _, rm, scores = model.forward(data, epoch_id)

                epoch_train_results_accuracy.append(accuracy_score(y_pred=np.argmax(out.data.cpu().numpy(),  axis=1), 
                                                                    y_true=labels))    
                epoch_train_results_f1_micro.append(f1_score(y_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                            y_true=labels, average='micro'))
                epoch_train_results_f1_macro.append(f1_score(y_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                            y_true=labels, average='macro'))
                epoch_train_results_nmi.append(normalized_mutual_info_score(labels_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                                            labels_true=labels))
                # rm = [m.data.cpu().numpy() for m in rm]
                # scores = [s.data.cpu().numpy() for s in scores]
                # epoch_ls_rm.append(rm)
                # epoch_ls_scores.append(scores)

            for idx, data in enumerate(val_loader):
                labels = data.y.data.numpy() 
                data = data.to(device)   
                out, _, _, rm, scores = model.forward(data, epoch_id)

                epoch_valid_results_accuracy.append(accuracy_score(y_pred=np.argmax(out.data.cpu().numpy(),  axis=1), 
                                                                    y_true=labels))    
                epoch_valid_results_f1_micro.append(f1_score(y_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                            y_true=labels, average='micro'))
                epoch_valid_results_f1_macro.append(f1_score(y_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                            y_true=labels, average='macro'))
                epoch_valid_results_nmi.append(normalized_mutual_info_score(labels_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                                            labels_true=labels))
                # rm = [m.data.cpu().numpy() for m in rm]
                # scores = [s.data.cpu().numpy() for s in scores]
                # epoch_ls_rm.append(rm)
                # epoch_ls_scores.append(scores)
                
            for idx, data in enumerate(test_loader):
                labels = data.y.data.numpy()
                data = data.to(device)
                out, _, _, rm, scores = model.forward(data, epoch_id)

                epoch_test_results_accuracy.append(accuracy_score(y_pred=np.argmax(out.data.cpu().numpy(),  axis=1), 
                                                                    y_true=labels))    
                epoch_test_results_f1_micro.append(f1_score(y_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                            y_true=labels, average='micro'))
                epoch_test_results_f1_macro.append(f1_score(y_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                            y_true=labels, average='macro'))
                epoch_test_results_nmi.append(normalized_mutual_info_score(labels_pred=np.argmax(out.data.cpu().numpy(), axis=1), 
                                                                            labels_true=labels))
                # rm = [m.data.cpu().numpy() for m in rm]
                # scores = [s.data.cpu().numpy() for s in scores]
                # epoch_ls_rm.append(rm)
                # epoch_ls_scores.append(scores)

            epoch_train_results_accuracy = np.mean(epoch_train_results_accuracy)
            epoch_train_results_f1_micro = np.mean(epoch_train_results_f1_micro)
            epoch_train_results_f1_macro = np.mean(epoch_train_results_f1_macro)
            epoch_train_results_nmi = np.mean(epoch_train_results_nmi)
            epoch_valid_results_accuracy = np.mean(epoch_valid_results_accuracy)
            epoch_valid_results_f1_micro = np.mean(epoch_valid_results_f1_micro)
            epoch_valid_results_f1_macro = np.mean(epoch_valid_results_f1_macro)
            epoch_valid_results_nmi = np.mean(epoch_valid_results_nmi)
            epoch_test_results_accuracy = np.mean(epoch_test_results_accuracy)
            epoch_test_results_f1_micro = np.mean(epoch_test_results_f1_micro)
            epoch_test_results_f1_macro = np.mean(epoch_test_results_f1_macro)
            epoch_test_results_nmi = np.mean(epoch_test_results_nmi)
            

            print('Evluating Epoch {}, time {:.3f}'.format(epoch_id, time.time()-start_epoch))
            print('train Accuracy = {:.4f}, valid Accuracy = {:.4f}, Test Accuracy = {:.4f}'.format(epoch_train_results_accuracy,
                                                                                                    epoch_valid_results_accuracy,
                                                                                                    epoch_test_results_accuracy))
            print('train Micro-F1 = {:.4f}, valid Micro-F1 = {:.4f}, Test Micro-F1 = {:.4f}'.format(epoch_train_results_f1_micro,
                                                                                                    epoch_valid_results_f1_micro,
                                                                                                    epoch_test_results_f1_micro))
            print('train Macro-F1 = {:.4f}, valid Macro-F1 = {:.4f}, Test Macro-F1 = {:.4f}'.format(epoch_train_results_f1_macro,
                                                                                                    epoch_valid_results_f1_macro,
                                                                                                    epoch_test_results_f1_macro))
            print('train NMI = {:.4f}, valid NMI = {:.4f}, Test NMI = {:.4f}'.format(epoch_train_results_nmi,
                                                                                    epoch_valid_results_nmi,
                                                                                    epoch_test_results_nmi))
            
            valid_results_accuracy.append(epoch_valid_results_accuracy)
            test_results_accuracy.append(epoch_test_results_accuracy)
            valid_results_f1_micro.append(epoch_valid_results_f1_micro)
            test_results_f1_micro.append(epoch_test_results_f1_micro)
            valid_results_f1_macro.append(epoch_valid_results_f1_macro)
            test_results_f1_macro.append(epoch_test_results_f1_macro)
            valid_results_nmi.append(epoch_valid_results_nmi)
            test_results_nmi.append(epoch_test_results_nmi)
            # ls_rm.append(epoch_ls_rm)
            # ls_scores.append(epoch_ls_scores)

            print('best valid Accuracy performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_accuracy),
                        test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))],\
                        model_config['verbose']*valid_results_accuracy.index(max(valid_results_accuracy))))
            print('best valid Micro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_f1_micro),
                        test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))],\
                        model_config['verbose']*valid_results_f1_micro.index(max(valid_results_f1_micro))))
            print('best valid Macro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_f1_macro),\
                        test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))],\
                        model_config['verbose']*valid_results_f1_macro.index(max(valid_results_f1_macro))))
            print('best valid NMI performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_nmi),\
                        test_results_nmi[valid_results_nmi.index(max(valid_results_nmi))],\
                        model_config['verbose']*valid_results_nmi.index(max(valid_results_nmi))))
            
            idx_1 = valid_results_accuracy.index(max(valid_results_accuracy))
            idx_2 = valid_results_f1_micro.index(max(valid_results_f1_micro))
            idx_3 = valid_results_f1_macro.index(max(valid_results_f1_macro))
            idx_4 = valid_results_nmi.index(max(valid_results_nmi))
            if (idx_1*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_2*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_3*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_4*model_config['verbose']+model_config['early_stop'] < epoch_id):
                break
    
    data_name = model_config['data_name']
    lr = model_config['adam_lr']
    batch_size = model_config['batch_size']
    local_agg_gnn = model_config['local_agg_gnn']
    gat_head = model_config['gat_head']
    fitness_mode = model_config['fitness_mode']
    pooling_mode = model_config['pooling_mode']
    output_mode = model_config['output_mode']
    num_levels = model_config['num_levels']
    hid_dim = model_config['hid_dim']
    overlap = model_config['overlap']
    all_cluster = model_config['all_cluster']
    pooling_ratio = model_config['pooling_ratio']
    cluster_range = model_config['cluster_range']
    drop_ratio = model_config['drop_ratio']
    loss_mode = model_config['loss_mode']

    now = datetime.datetime.now()
    path = f"../output/GC_{data_name}_model_lr_{lr}_{batch_size}"+\
            f"_gnn_{local_agg_gnn}_{gat_head}_fit_{fitness_mode}"+\
            f"_out_{output_mode}"+\
            f"_lev_{num_levels}_hid_{hid_dim}_ol_{overlap}_all_{str(all_cluster)}"+\
            f"_po_{pooling_mode}_{pooling_ratio}_cluster_{cluster_range}_drop_{drop_ratio}_{loss_mode}.dat"
    print(path)
    with open(path,"a+") as f:
        f.write('\n'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+
                str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+str(now.microsecond)+' '+
                str(test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))])+' '+
                str(test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))])+' '+
                str(test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))])+' '+
                str(test_results_nmi[valid_results_nmi.index(max(valid_results_nmi))])+' '+
                str(SEED))

    # rm = ls_rm[valid_results_accuracy.index(max(valid_results_accuracy))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_rm.txt', rm)
    
    # scores = ls_scores[valid_results_accuracy.index(max(valid_results_accuracy))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_sco.txt', scores)

    # torch.save(train_loader.dataset, path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_train.pt')
    # torch.save(val_loader.dataset, path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_val.pt')
    # torch.save(test_loader.dataset, path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_test.pt')


def NC_run_dphi(model, optimizer, ls_data, model_config, device, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    writer = SummaryWriter(logdir='runs/{}'.format(model_config['alias']))
    writer.add_text('config', str(model_config), 0)

    # loss_func = torch.nn.NLLLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()

    test_results_accuracy = []
    valid_results_accuracy = []
    test_results_f1_micro = []
    valid_results_f1_micro = []
    test_results_f1_macro = []
    valid_results_f1_macro = []
    ls_rm = []
    ls_scores = []

    for epoch_id, epoch in enumerate(range(model_config['num_epoch'])):
        start_epoch = time.time()
        if epoch_id % model_config['verbose'] == 0:
            print('Epoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss = 0

        for idx, data in enumerate(ls_data):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            labels_tensor = data.y

            model.train()
            optimizer.zero_grad()
            out, loss_recon, loss_kl, _, _ = model.forward(data, epoch_id)
            preds_train = out[train_mask]
            loss_nc = loss_func(preds_train, labels_tensor[train_mask])

            if model_config['loss_mode']=='all':
                loss = loss_nc + 0.1 * loss_recon + 1 * loss_kl
            elif model_config['loss_mode']=='kl':
                loss = loss_nc + 1 * loss_kl
            elif model_config['loss_mode']=='recon':
                loss = loss_nc + 0.1 * loss_recon
            elif model_config['loss_mode']=='nc':
                loss = loss_nc
            print(loss_nc.data.cpu().numpy().tolist(), loss_recon.data.cpu().numpy().tolist(), loss_kl.data.cpu().numpy().tolist())
            print(loss.data.cpu().numpy().tolist())

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().item()

        # write epoch info
        writer.add_scalar('model/loss', total_loss, epoch_id)

        # evaluate epoch
        model.eval()
        for idx, data in enumerate(ls_data):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            labels = data.y.data.cpu().numpy()

            out, _, _, rm, scores = model.forward(data, epoch_id)
            out = (out > 0).float().cpu()

            preds_train = out[train_mask]
            preds_valid = out[val_mask]
            preds_test = out[test_mask]

        if epoch_id % model_config['verbose'] == 0:
            epoch_train_results_accuracy = accuracy_score(y_pred=preds_train.data.cpu().numpy(), 
                                                        y_true=labels[train_mask.data.cpu().numpy()])
            epoch_valid_results_accuracy = accuracy_score(y_pred=preds_valid.data.cpu().numpy(), 
                                                        y_true=labels[val_mask.data.cpu().numpy()])
            epoch_test_results_accuracy = accuracy_score(y_pred=preds_test.data.cpu().numpy(), 
                                                        y_true=labels[test_mask.data.cpu().numpy()])

            epoch_train_results_f1_micro = f1_score(y_pred=preds_train.data.cpu().numpy(), 
                                                    y_true=labels[train_mask.data.cpu().numpy()], average='micro')
            epoch_valid_results_f1_micro = f1_score(y_pred=preds_valid.data.cpu().numpy(), 
                                                    y_true=labels[val_mask.data.cpu().numpy()], average='micro')
            epoch_test_results_f1_micro = f1_score(y_pred=preds_test.data.cpu().numpy(), 
                                                y_true=labels[test_mask.data.cpu().numpy()], average='micro')

            epoch_train_results_f1_macro = f1_score(y_pred=preds_train.data.cpu().numpy(), 
                                                    y_true=labels[train_mask.data.cpu().numpy()], average='macro')
            epoch_valid_results_f1_macro = f1_score(y_pred=preds_valid.data.cpu().numpy(), 
                                                    y_true=labels[val_mask.data.cpu().numpy()], average='macro')
            epoch_test_results_f1_macro = f1_score(y_pred=preds_test.data.cpu().numpy(), 
                                                    y_true=labels[test_mask.data.cpu().numpy()], average='macro')

            print('Evluating Epoch {}, time {:.3f}'.format(epoch_id, time.time()-start_epoch))
            print('train Accuracy = {:.4f}, valid Accuracy = {:.4f}, Test Accuracy = {:.4f}'.format(epoch_train_results_accuracy,
                                                                                                    epoch_valid_results_accuracy,
                                                                                                    epoch_test_results_accuracy))
            print('train Micro-F1 = {:.4f}, valid Micro-F1 = {:.4f}, Test Micro-F1 = {:.4f}'.format(epoch_train_results_f1_micro,
                                                                                                    epoch_valid_results_f1_micro,
                                                                                                    epoch_test_results_f1_micro))
            print('train Macro-F1 = {:.4f}, valid Macro-F1 = {:.4f}, Test Macro-F1 = {:.4f}'.format(epoch_train_results_f1_macro,
                                                                                                    epoch_valid_results_f1_macro,
                                                                                                    epoch_test_results_f1_macro))
            
            valid_results_accuracy.append(epoch_valid_results_accuracy)
            test_results_accuracy.append(epoch_test_results_accuracy)
            valid_results_f1_micro.append(epoch_valid_results_f1_micro)
            test_results_f1_micro.append(epoch_test_results_f1_micro)
            valid_results_f1_macro.append(epoch_valid_results_f1_macro)
            test_results_f1_macro.append(epoch_test_results_f1_macro)
            
            rm = [m.data.cpu().numpy() for m in rm]
            scores = [s.data.cpu().numpy() for s in scores]
            ls_rm.append(rm)
            ls_scores.append(scores)

            print('best valid Accuracy performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_accuracy),
                        test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))],\
                        model_config['verbose']*valid_results_accuracy.index(max(valid_results_accuracy))))
            print('best valid Micro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_f1_micro),
                        test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))],\
                        model_config['verbose']*valid_results_f1_micro.index(max(valid_results_f1_micro))))
            print('best valid Macro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_f1_macro),\
                        test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))],\
                        model_config['verbose']*valid_results_f1_macro.index(max(valid_results_f1_macro))))
            
            idx_1 = valid_results_accuracy.index(max(valid_results_accuracy))
            idx_2 = valid_results_f1_micro.index(max(valid_results_f1_micro))
            idx_3 = valid_results_f1_macro.index(max(valid_results_f1_macro))
            if (idx_1*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_2*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_3*model_config['verbose']+model_config['early_stop'] < epoch_id):
                break

    data_name = model_config['data_name']
    lr = model_config['adam_lr']
    num_train = model_config['num_train']
    num_class = model_config['num_class']
    local_agg_gnn = model_config['local_agg_gnn']
    gat_head = model_config['gat_head']
    fitness_mode = model_config['fitness_mode']
    pooling_mode = model_config['pooling_mode']
    output_mode = model_config['output_mode']
    num_levels = model_config['num_levels']
    hid_dim = model_config['hid_dim']
    overlap = model_config['overlap']
    all_cluster = model_config['all_cluster']
    pooling_ratio = model_config['pooling_ratio']
    cluster_range = model_config['cluster_range']
    drop_ratio = model_config['drop_ratio']
    loss_mode = model_config['loss_mode']

    now = datetime.datetime.now()
    path = f"../output/NC_{data_name}_model_lr_{lr}_n_train_{num_train}_n_class_{num_class}"+\
            f"_gnn_{local_agg_gnn}_{gat_head}_fit_{fitness_mode}"+\
            f"_out_{output_mode}"+\
            f"_lev_{num_levels}_hid_{hid_dim}_ol_{overlap}_all_{str(all_cluster)}"+\
            f"_pool_{pooling_mode}_{pooling_ratio}_cluster_{cluster_range}_drop_{drop_ratio}_{loss_mode}.dat"
    print(path)
    with open(path,"a+") as f:
        f.write('\n'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+
                str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+str(now.microsecond)+' '+
                str(test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))])+' '+
                str(test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))])+' '+
                str(test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))])+' '+
                str(SEED))

    # rm = ls_rm[valid_results_accuracy.index(max(valid_results_accuracy))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_rm.txt', rm)

    # scores = ls_scores[valid_results_accuracy.index(max(valid_results_accuracy))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_sco.txt', scores)

    # torch.save(ls_data[0], path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'.pt')

def NC_run(model, optimizer, ls_data, model_config, device, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    writer = SummaryWriter(logdir='runs/{}'.format(model_config['alias']))
    writer.add_text('config', str(model_config), 0)

    loss_func = torch.nn.NLLLoss()

    test_results_accuracy = []
    valid_results_accuracy = []
    test_results_f1_micro = []
    valid_results_f1_micro = []
    test_results_f1_macro = []
    valid_results_f1_macro = []
    test_results_nmi = []
    valid_results_nmi = []
    ls_rm = []
    ls_scores = []
    ls_fitness = []

    for epoch_id, epoch in enumerate(range(model_config['num_epoch'])):
        start_epoch = time.time()
        if epoch_id % model_config['verbose'] == 0:
            print('Epoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss = 0
        
        for idx, data in enumerate(ls_data):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            labels_tensor = data.y
            
            model.train()
            optimizer.zero_grad()
            out, loss_recon, loss_kl, _, _, _ = model.forward(data, epoch_id)
            preds_train = out[train_mask]
            loss_nc = loss_func(preds_train, labels_tensor[train_mask])
            
            if model_config['loss_mode']=='all':
                loss = loss_nc + 0.1 * loss_recon + 1 * loss_kl
            elif model_config['loss_mode']=='kl':
                loss = loss_nc + 1 * loss_kl
            elif model_config['loss_mode']=='recon':
                loss = loss_nc + 0.1 * loss_recon
            elif model_config['loss_mode']=='nc':
                loss = loss_nc
            print(loss_nc.data.cpu().numpy().tolist(), loss_recon.data.cpu().numpy().tolist(), loss_kl.data.cpu().numpy().tolist())
            print(loss.data.cpu().numpy().tolist())

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().item()

        # write epoch info
        writer.add_scalar('model/loss', total_loss, epoch_id)
        
        # evaluate epoch
        model.eval()
        for idx, data in enumerate(ls_data):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            labels = data.y.data.cpu().numpy()
            
            out, _, _, rm, scores, fitness = model.forward(data, epoch_id)

            preds_train = out[train_mask]
            preds_valid = out[val_mask]
            preds_test = out[test_mask]
            
        if epoch_id % model_config['verbose'] == 0:
            epoch_train_results_accuracy = accuracy_score(y_pred=np.argmax(preds_train.data.cpu().numpy(),  axis=1), 
                                                        y_true=labels[train_mask.data.cpu().numpy()])
            epoch_valid_results_accuracy = accuracy_score(y_pred=np.argmax(preds_valid.data.cpu().numpy(),  axis=1), 
                                                        y_true=labels[val_mask.data.cpu().numpy()])
            epoch_test_results_accuracy = accuracy_score(y_pred=np.argmax(preds_test.data.cpu().numpy(),  axis=1), 
                                                        y_true=labels[test_mask.data.cpu().numpy()])
            
            epoch_train_results_f1_micro = f1_score(y_pred=np.argmax(preds_train.data.cpu().numpy(), axis=1), 
                                                    y_true=labels[train_mask.data.cpu().numpy()], average='micro')
            epoch_valid_results_f1_micro = f1_score(y_pred=np.argmax(preds_valid.data.cpu().numpy(), axis=1), 
                                                    y_true=labels[val_mask.data.cpu().numpy()], average='micro')
            epoch_test_results_f1_micro = f1_score(y_pred=np.argmax(preds_test.data.cpu().numpy(), axis=1), 
                                                y_true=labels[test_mask.data.cpu().numpy()], average='micro')
            
            epoch_train_results_f1_macro = f1_score(y_pred=np.argmax(preds_train.data.cpu().numpy(), axis=1), 
                                                    y_true=labels[train_mask.data.cpu().numpy()], average='macro')
            epoch_valid_results_f1_macro = f1_score(y_pred=np.argmax(preds_valid.data.cpu().numpy(), axis=1), 
                                                    y_true=labels[val_mask.data.cpu().numpy()], average='macro')
            epoch_test_results_f1_macro = f1_score(y_pred=np.argmax(preds_test.data.cpu().numpy(), axis=1), 
                                                    y_true=labels[test_mask.data.cpu().numpy()], average='macro')
            
            epoch_train_results_nmi = normalized_mutual_info_score(labels_pred=np.argmax(preds_train.data.cpu().numpy(), axis=1), 
                                                                labels_true=labels[train_mask.data.cpu().numpy()])
            epoch_valid_results_nmi = normalized_mutual_info_score(labels_pred=np.argmax(preds_valid.data.cpu().numpy(), axis=1), 
                                                                labels_true=labels[val_mask.data.cpu().numpy()])
            epoch_test_results_nmi = normalized_mutual_info_score(labels_pred=np.argmax(preds_test.data.cpu().numpy(), axis=1), 
                                                                labels_true=labels[test_mask.data.cpu().numpy()])
            
            print('Evluating Epoch {}, time {:.3f}'.format(epoch_id, time.time()-start_epoch))
            print('train Accuracy = {:.4f}, valid Accuracy = {:.4f}, Test Accuracy = {:.4f}'.format(epoch_train_results_accuracy,
                                                                                                    epoch_valid_results_accuracy,
                                                                                                    epoch_test_results_accuracy))
            print('train Micro-F1 = {:.4f}, valid Micro-F1 = {:.4f}, Test Micro-F1 = {:.4f}'.format(epoch_train_results_f1_micro,
                                                                                                    epoch_valid_results_f1_micro,
                                                                                                    epoch_test_results_f1_micro))
            print('train Macro-F1 = {:.4f}, valid Macro-F1 = {:.4f}, Test Macro-F1 = {:.4f}'.format(epoch_train_results_f1_macro,
                                                                                                    epoch_valid_results_f1_macro,
                                                                                                    epoch_test_results_f1_macro))
            print('train NMI = {:.4f}, valid NMI = {:.4f}, Test NMI = {:.4f}'.format(epoch_train_results_nmi,
                                                                                    epoch_valid_results_nmi,
                                                                                    epoch_test_results_nmi))
            
            valid_results_accuracy.append(epoch_valid_results_accuracy)
            test_results_accuracy.append(epoch_test_results_accuracy)
            valid_results_f1_micro.append(epoch_valid_results_f1_micro)
            test_results_f1_micro.append(epoch_test_results_f1_micro)
            valid_results_f1_macro.append(epoch_valid_results_f1_macro)
            test_results_f1_macro.append(epoch_test_results_f1_macro)
            valid_results_nmi.append(epoch_valid_results_nmi)
            test_results_nmi.append(epoch_test_results_nmi)

            # rm = [m.data.cpu().numpy() for m in rm]
            # scores = [s.data.cpu().numpy() for s in scores]
            # ls_rm.append(rm)
            # ls_scores.append(scores)

            fitness = [fit.data.cpu().numpy() for fit in fitness]
            ls_fitness.append(fitness)

            print('best valid Accuracy performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_accuracy),
                        test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))],\
                        model_config['verbose']*valid_results_accuracy.index(max(valid_results_accuracy))))
            print('best valid Micro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_f1_micro),
                        test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))],\
                        model_config['verbose']*valid_results_f1_micro.index(max(valid_results_f1_micro))))
            print('best valid Macro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_f1_macro),\
                        test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))],\
                        model_config['verbose']*valid_results_f1_macro.index(max(valid_results_f1_macro))))
            print('best valid NMI performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_nmi),\
                        test_results_nmi[valid_results_nmi.index(max(valid_results_nmi))],\
                        model_config['verbose']*valid_results_nmi.index(max(valid_results_nmi))))
            
            idx_1 = valid_results_accuracy.index(max(valid_results_accuracy))
            idx_2 = valid_results_f1_micro.index(max(valid_results_f1_micro))
            idx_3 = valid_results_f1_macro.index(max(valid_results_f1_macro))
            idx_4 = valid_results_nmi.index(max(valid_results_nmi))
            if (idx_1*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_2*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_3*model_config['verbose']+model_config['early_stop'] < epoch_id)&(idx_4*model_config['verbose']+model_config['early_stop'] < epoch_id):
                break
    
    data_name = model_config['data_name']
    lr = model_config['adam_lr']
    num_train = model_config['num_train']
    local_agg_gnn = model_config['local_agg_gnn']
    gat_head = model_config['gat_head']
    fitness_mode = model_config['fitness_mode']
    pooling_mode = model_config['pooling_mode']
    output_mode = model_config['output_mode']
    num_levels = model_config['num_levels']
    hid_dim = model_config['hid_dim']
    overlap = model_config['overlap']
    all_cluster = model_config['all_cluster']
    pooling_ratio = model_config['pooling_ratio']
    cluster_range = model_config['cluster_range']
    drop_ratio = model_config['drop_ratio']
    loss_mode = model_config['loss_mode']

    now = datetime.datetime.now()
    path = f"../output/NC_{data_name}_model_lr_{lr}_n_train_{num_train}"+\
            f"_gnn_{local_agg_gnn}_{gat_head}_fit_{fitness_mode}"+\
            f"_out_{output_mode}"+\
            f"_lev_{num_levels}_hid_{hid_dim}_ol_{overlap}_all_{str(all_cluster)}"+\
            f"_pool_{pooling_mode}_{pooling_ratio}_cluster_{cluster_range}_drop_{drop_ratio}_{loss_mode}.dat"
    print(path)
    with open(path,"a+") as f:
        f.write('\n'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+
                str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+str(now.microsecond)+' '+
                str(test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))])+' '+
                str(test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))])+' '+
                str(test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))])+' '+
                str(test_results_nmi[valid_results_nmi.index(max(valid_results_nmi))])+' '+
                str(SEED))
    
    # rm = ls_rm[valid_results_accuracy.index(max(valid_results_accuracy))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_rm.txt', rm)

    # scores = ls_scores[valid_results_accuracy.index(max(valid_results_accuracy))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_sco.txt', scores)

    # torch.save(ls_data[0], path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'.pt')

    fitness = ls_fitness[valid_results_accuracy.index(max(valid_results_accuracy))]
    np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_fit.txt', fitness)


def LP_run(model, optimizer, ls_data, model_config, n_clusters, device, SEED):
    writer = SummaryWriter(logdir='runs/{}'.format(model_config['alias']))
    writer.add_text('config', str(model_config), 0)

    test_results_auc = []
    valid_results_auc = []
    test_results_ap = []
    valid_results_ap = []
    ls_rm = []
    ls_scores = []

    for epoch_id, epoch in enumerate(range(model_config['num_epoch'])):
        start_epoch = time.time()
        if epoch_id % model_config['verbose'] == 0:
            print('\nEpoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss = 0
        
        for idx, data in enumerate(ls_data):
            model.train()
            optimizer.zero_grad()
            z, loss_kl, _, _ = model.encode(data, epoch_id)
            loss_recon = model.recon_loss(z, data.train_pos_edge_index)

            if model_config['loss_mode']=='kl':
                loss = loss_recon + loss_kl
            elif model_config['loss_mode']=='lp':
                loss = loss_recon
            print(loss_recon.data.cpu().numpy().tolist(), loss_kl.data.cpu().numpy().tolist())
            print(loss.data.cpu().numpy().tolist())

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().item()

        # write epoch info
        writer.add_scalar('model/loss', total_loss, epoch_id)
        
        # evaluate epoch
        model.eval()
        for idx, data in enumerate(ls_data):
            z, _, rm, scores = model.encode(data, epoch_id)
            
            # train_auc, train_ap = model.test(z, data.train_pos_edge_index, data.train_neg_edge_index)
            train_auc, train_ap = 0, 0
            val_auc, val_ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
            test_auc, test_ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

        if epoch_id % model_config['verbose'] == 0:
            print('Evluating Epoch {}, time {:.3f}'.format(epoch_id, time.time()-start_epoch))
            print('train ROC-AUC = {:.4f}, valid ROC-AUC = {:.4f}, Test ROC-AUC = {:.4f}'.format(train_auc, val_auc, test_auc))
            print('train AP = {:.4f}, valid AP = {:.4f}, Test AP = {:.4f}'.format(train_ap, val_ap, test_ap))
            
            valid_results_auc.append(val_auc)
            valid_results_ap.append(val_ap)
            test_results_auc.append(test_auc)
            test_results_ap.append(test_ap)

            rm = [m.data.cpu().numpy() for m in rm]
            scores = [s.data.cpu().numpy() for s in scores]
            ls_rm.append(rm)
            ls_scores.append(scores)
            
            print('best valid AUC performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_auc),
                        test_results_auc[valid_results_auc.index(max(valid_results_auc))],\
                        model_config['verbose']*valid_results_auc.index(max(valid_results_auc))))
            print('best valid AP performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.\
                format(max(valid_results_ap),
                        test_results_ap[valid_results_ap.index(max(valid_results_ap))],\
                        model_config['verbose']*valid_results_ap.index(max(valid_results_ap))))
            
            idx_1 = valid_results_auc.index(max(valid_results_auc))
            idx_2 = valid_results_ap.index(max(valid_results_ap))
            if (idx_1*model_config['verbose']+model_config['early_stop'] < epoch_id) and (idx_2*model_config['verbose']+model_config['early_stop'] < epoch_id):
                break
    
    data_name = model_config['data_name']
    adam_lr = model_config['adam_lr']
    ratio_train = model_config['ratio_train']
    local_agg_gnn = model_config['local_agg_gnn']
    gat_head = model_config['gat_head']
    fitness_mode = model_config['fitness_mode']
    pooling_mode = model_config['pooling_mode']
    num_levels = model_config['num_levels']
    hid_dim = model_config['hid_dim']
    overlap = model_config['overlap']
    all_cluster = model_config['all_cluster']
    pooling_ratio = model_config['pooling_ratio']
    cluster_range = model_config['cluster_range']
    drop_ratio = model_config['drop_ratio']
    loss_mode = model_config['loss_mode']
    
    now = datetime.datetime.now()
    path = f"../output/LP_{data_name}_model_lr_{adam_lr}_train_{ratio_train}"+\
            f"_gnn_{local_agg_gnn}_{gat_head}_fit_{fitness_mode}"+\
            f"_lev_{num_levels}_hid_{hid_dim}_ol_{overlap}_all_{str(all_cluster)}"+\
            f"_pool_{pooling_mode}_{pooling_ratio}_cluster_{cluster_range}_drop_{drop_ratio}_{loss_mode}.dat"
    print(path)
    with open(path,"a+") as f:
        f.write('\n'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+
                str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+str(now.microsecond)+' '+
                str(test_results_auc[valid_results_auc.index(max(valid_results_auc))])+' '+
                str(test_results_ap[valid_results_ap.index(max(valid_results_ap))])+' '+
                str(SEED))
    
    # rm_1 = ls_rm[valid_results_auc.index(max(valid_results_auc))]
    # rm_2 = ls_rm[valid_results_ap.index(max(valid_results_ap))]
    # print(sum(rm_1[-1].sum(0)>1), sum(rm_2[-1].sum(0)>1))
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_rm_1.txt', rm_1)
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_rm_2.txt', rm_2)

    # scores_1 = ls_scores[valid_results_auc.index(max(valid_results_auc))]
    # scores_2 = ls_scores[valid_results_ap.index(max(valid_results_ap))]
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_sco_1.txt', scores_1)
    # np.save(path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'_sco_2.txt', scores_2)

    # torch.save(ls_data[0], path[:-4]+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)+'.pt')
