import numpy as np
import random
import time
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F


def GC_run(args, model, optimizer, train_loader, val_loader, test_loader):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loss_func = torch.nn.NLLLoss()

    test_accuracy = []
    valid_accuracy = []

    for epoch_id, epoch in enumerate(range(args.num_epoch)):
        start_epoch = time.time()
        if epoch_id % args.verbose == 0:
            print('Epoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss_recon = 0
        total_loss_kl = 0
        total_loss_nc = 0
        total_loss = 0

        for idx, data in enumerate(train_loader):
            data = data.to(args.device)
            labels_tensor = data.y

            model.train()
            optimizer.zero_grad()
            embedding, loss_recon, loss_kl, _, _ = model.forward(data)
            out = F.log_softmax(embedding, dim=1)
            loss_nc = loss_func(out, labels_tensor)

            if args.loss_mode == 'all':
                loss = loss_nc + 0.1 * loss_recon + 1 * loss_kl  # 1 10 100 1000
            elif args.loss_mode == 'kl':
                loss = loss_nc + 1 * loss_kl
            elif args.loss_mode == 'recon':
                loss = loss_nc + 0.1 * loss_recon
            elif args.loss_mode == 'nc':
                loss = loss_nc
            else:
                loss = None

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

        # evaluate epoch
        if epoch_id % args.verbose == 0:
            model.eval()
            epoch_train_accuracy = []
            epoch_valid_accuracy = []
            epoch_test_accuracy = []

            for idx, data in enumerate(train_loader):
                labels = data.y.data.numpy()
                data = data.to(args.device)
                out, _, _, _, _ = model.forward(data)
                pred = np.argmax(out.data.cpu().numpy(), axis=1)

                epoch_train_accuracy.append(
                    accuracy_score(y_pred=pred, y_true=labels)
                )

            for idx, data in enumerate(val_loader):
                labels = data.y.data.numpy()
                data = data.to(args.device)
                out, _, _, _, scores = model.forward(data)
                pred = np.argmax(out.data.cpu().numpy(), axis=1)

                epoch_valid_accuracy.append(
                    accuracy_score(y_pred=pred, y_true=labels)
                )

            for idx, data in enumerate(test_loader):
                labels = data.y.data.numpy()
                data = data.to(args.device)
                out, _, _, _, scores = model.forward(data)
                pred = np.argmax(out.data.cpu().numpy(), axis=1)

                epoch_test_accuracy.append(
                    accuracy_score(y_pred=pred, y_true=labels)
                )

            epoch_train_accuracy = np.mean(epoch_train_accuracy)
            epoch_valid_accuracy = np.mean(epoch_valid_accuracy)
            epoch_test_accuracy = np.mean(epoch_test_accuracy)
            print('Evaluating Epoch {}, time {:.3f}'.format(epoch_id, time.time() - start_epoch))
            print('train Accuracy = {:.4f}, valid Accuracy = {:.4f}, Test Accuracy = {:.4f}'.format(
                epoch_train_accuracy,
                epoch_valid_accuracy,
                epoch_test_accuracy
            ))

            valid_accuracy.append(epoch_valid_accuracy)
            test_accuracy.append(epoch_test_accuracy)
            print('best valid Accuracy is {:.4f}, best test is {:.4f} and epoch_id is {}'.format(
                max(valid_accuracy),
                test_accuracy[valid_accuracy.index(max(valid_accuracy))],
                args.verbose * valid_accuracy.index(max(valid_accuracy))
            ))

            # early stop
            idx_1 = valid_accuracy.index(max(valid_accuracy))
            if (idx_1 * args.verbose + args.early_stop < epoch_id):
                break


def NC_run(args, model, optimizer, data):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loss_func = torch.nn.NLLLoss()
    valid_f1_micro = []
    test_f1_micro = []

    for epoch_id, epoch in enumerate(range(args.num_epoch)):
        start_epoch = time.time()
        if epoch_id % args.verbose == 0:
            print('\n Epoch {} starts !'.format(epoch_id))
            print('-' * 80)

        train_mask = data.train_mask
        labels_tensor = data.y

        model.train()
        optimizer.zero_grad()
        embedding, loss_recon, loss_kl, _, _ = model.forward(data)
        out = F.log_softmax(embedding, dim=1)
        loss_nc = loss_func(out[train_mask], labels_tensor[train_mask])

        if args.loss_mode == 'all':
            loss = loss_nc + 0.1 * loss_recon + 1 * loss_kl
        elif args.loss_mode == 'kl':
            loss = loss_nc + 1 * loss_kl
        elif args.loss_mode == 'recon':
            loss = loss_nc + 0.1 * loss_recon
        elif args.loss_mode == 'nc':
            loss = loss_nc
        else:
            loss = None
        print('Loss values: ',
              round(loss_nc.data.cpu().numpy().tolist(), 5),
              round(loss_recon.data.cpu().numpy().tolist(), 5),
              round(loss_kl.data.cpu().numpy().tolist(), 5))
        print('Final loss: ', round(loss.data.cpu().numpy().tolist(), 5))

        # update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # evaluate epoch
        model.eval()
        train_mask = data.train_mask.data.cpu().numpy()
        val_mask = data.val_mask.data.cpu().numpy()
        test_mask = data.test_mask.data.cpu().numpy()
        labels = data.y.data.cpu().numpy()

        embedding, _, _, _, index = model.forward(data)
        out = F.log_softmax(embedding, dim=1)
        pred = np.argmax(out.data.cpu().numpy(), axis=1)

        pred_train, labels_train = pred[train_mask], labels[train_mask]
        pred_valid, labels_valid = pred[val_mask], labels[val_mask]
        pred_test, labels_test = pred[test_mask], labels[test_mask]

        if epoch_id % args.verbose == 0:
            epoch_train_f1_micro = f1_score(
                y_pred=pred_train, y_true=labels_train, average='micro'
            )
            epoch_valid_f1_micro = f1_score(
                y_pred=pred_valid, y_true=labels_valid, average='micro'
            )
            epoch_test_f1_micro = f1_score(
                y_pred=pred_test, y_true=labels_test, average='micro'
            )

            print('Evaluating Epoch {}, time {:.3f}'.format(epoch_id, time.time() - start_epoch))
            print('train Micro-F1 = {:.4f}, valid Micro-F1 = {:.4f}, Test Micro-F1 = {:.4f}'.format(
                epoch_train_f1_micro, epoch_valid_f1_micro, epoch_test_f1_micro
            ))
            valid_f1_micro.append(epoch_valid_f1_micro)
            test_f1_micro.append(epoch_test_f1_micro)
            print('best valid Micro-F1 is {:.4f}, best test is {:.4f} and epoch_id is {}'.format(
                max(valid_f1_micro),
                test_f1_micro[valid_f1_micro.index(max(valid_f1_micro))],
                args.verbose * valid_f1_micro.index(max(valid_f1_micro))
            ))

            # early stop
            idx_1 = valid_f1_micro.index(max(valid_f1_micro))
            if (idx_1 * args.verbose + args.early_stop < epoch_id):
                break


def LP_run(args, model, optimizer, data):
    valid_auc = []
    test_auc = []

    for epoch_id, epoch in enumerate(range(args.num_epoch)):
        start_epoch = time.time()
        if epoch_id % args.verbose == 0:
            print('\nEpoch {} starts !'.format(epoch_id))
            print('-' * 80)

        model.train()
        optimizer.zero_grad()
        z, _, loss_kl, _, _ = model.encode(data)
        loss_recon = model.recon_loss(z, data.train_pos_edge_index)

        if args.loss_mode == 'kl':
            loss = loss_recon + loss_kl
        elif args.loss_mode == 'lp':
            loss = loss_recon
        else:
            loss = None
        print(loss_recon.data.cpu().numpy().tolist(), loss_kl.data.cpu().numpy().tolist())
        print(loss.data.cpu().numpy().tolist())

        # update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # evaluate epoch
        model.eval()
        z, _, _, _, _ = model.encode(data)

        epoch_train_auc, _ = model.test(z, data.train_pos_edge_index, data.train_neg_edge_index)
        epoch_val_auc, _ = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
        epoch_test_auc, _ = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

        if epoch_id % args.verbose == 0:
            print('Evaluating Epoch {}, time {:.3f}'.format(epoch_id, time.time() - start_epoch))
            print('train ROC-AUC = {:.4f}, valid ROC-AUC = {:.4f}, Test ROC-AUC = {:.4f}'.format(
                epoch_train_auc, epoch_val_auc, epoch_test_auc
            ))
            valid_auc.append(epoch_val_auc)
            test_auc.append(epoch_test_auc)

            print('best valid AUC performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.format(
                max(valid_auc),
                test_auc[valid_auc.index(max(valid_auc))],
                args.verbose * valid_auc.index(max(valid_auc))
            ))

            # early stop
            idx_1 = valid_auc.index(max(valid_auc))
            if (idx_1 * args.verbose + args.early_stop < epoch_id):
                break
