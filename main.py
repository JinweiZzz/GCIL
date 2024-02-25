# -*- coding: utf-8 -*-
"""
@author: zengjinwei

"""

import os
import random
import time
import setproctitle
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from utils import mean_absolute_percentage_error

from utils import create_dataset, split_train_val_test, make_batch
from models import Treatment, Outcome, Resnet, Densenet, MLP
import mlflow
from mlflow.tracking import MlflowClient
import logging

import warnings

warnings.filterwarnings("ignore")


def init_model(args):
    '''
    Define and initialize model, optimizer, and scheduler
    '''
    global classifier, optimizer_c
    if args.disentangle == 1:
        model = Outcome(args).to(args.device)
        classifier = Treatment(args).to(args.device)
        optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lra, weight_decay=args.weight_decay)
        mlflow.log_param('param_num_c', sum(p.numel() for p in classifier.parameters()))
    if args.disentangle != 1:
        if args.base == 'Resnet':
            model = Resnet(args).to(args.device)
        if args.base == 'MLP':
            model = MLP(args).to(args.device)
        if args.base == 'Densenet':
            model = Densenet(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    print('#Trainable Parameters:', np.sum([p.numel() for p in train_params]))
    mlflow.log_param('param_num', sum(p.numel() for p in model.parameters()))
    if args.disentangle == 1:
        return model, optimizer, classifier, optimizer_c
    else:
        return model, optimizer


def loss_treatment_prediction(ids, predicted_treatment, treatment):
    return torch.nn.L1Loss(reduction='none')(predicted_treatment[ids, :].float(), treatment[ids, :].float())


def loss_outcome_prediction(args, ids, predicted_outcome, outcome):
    if args.classify == 1:
        return F.binary_cross_entropy(predicted_outcome[ids, :], outcome[ids, :], reduction='none')
    else:   # regression
        return torch.nn.L1Loss(reduction='none')(predicted_outcome[ids, :].float(), outcome[ids, :].float())


def pretrain(args, data, train_loaders, val):
    start = time.time()
    classifier.train()
    data = data.to(args.device)
    val = val.to(args.device)
    factual_treatment = data.t.view(-1, 1)
    x = data.x

    n_step = 0
    val_pretrain_loss = 1e5
    for epoch in range(args.pretrain_epochs):
        loss_log = 0.
        train_ids_all = []
        n_train = 0
        for batch_idx, train_ids in enumerate(train_loaders):
            train_ids_all += list(train_ids)
            n_train += len(train_ids)
            optimizer_c.zero_grad()

            din = x.view(-1, args.n_features)
            predicted_treatment = classifier(args, din)
            treatment_loss = loss_treatment_prediction(train_ids, predicted_treatment, factual_treatment)
            loss = treatment_loss

            loss = torch.sum(loss)
            loss_log += loss.item()
            loss.backward()
            optimizer_c.step()
            time_iter = time.time() - start
        loss_log = loss_log / n_train
        if epoch % 10 == 0:
            print('Epoch {}:'.format(epoch))
            print("FOLD {}, Time {:.4f} -- Pretrain Treatment prediction training mae:{}".format(fold, time_iter, loss_log))
        mlflow.log_metric(key='pretrain_epoch', value=epoch, step=n_step)
        mlflow.log_metric(key='pretrain_mae_loss', value=loss_log, step=n_step)
        val_mae_loss = test_treatment(val, 'Pretrain')
        if epoch % 10 == 0:
            print("FOLD {}, Time {:.4f} -- Pretrain Treatment prediction validation mae:{}".format(fold, time_iter, val_mae_loss))
        mlflow.log_metric(key='val_mae_loss', value=val_mae_loss, step=n_step)
        if val_mae_loss < val_pretrain_loss:
            print("Epoch {}:  Pretrain Model Saved !!!!!!!!!!".format(epoch))
            torch.save(classifier.state_dict(), 'model/pretrain_model_classifier_{}_{}'.format(
                args.exp_name_c, args.model_name_suffix))
            val_pretrain_loss = val_mae_loss
        n_step += 1


def train_process(args, data, train_loaders, val):
    if args.disentangle == 1:
        classifier.load_state_dict(
            torch.load('model/pretrain_model_classifier_{}_{}'.format(
                args.exp_name_c, args.model_name_suffix)))
        classifier.train()
    model.train()
    start = time.time()
    min_loss = 1e5
    max_auc = 0
    patience = 0
    n_step = 0
    t_step = 0
    data = data.to(args.device)
    factual_outcome = data.outcome.view(-1, 1)
    factual_treatment = data.t.view(-1, 1)
    for epoch in range(args.epochs):
        loss_log = 0.
        loss_log_t = 0.
        train_ids_all = []
        n_train = 0
        if args.classify == 1:
            for batch_idx, train_ids in enumerate(train_loaders):
                train_ids_all += list(train_ids)
                n_train += len(train_ids)
                optimizer.zero_grad()
                if args.disentangle == 1:
                    embedding, predicted_outcome = model(args, data)
                    predicted_treatment = classifier(args, embedding)
                    outcome_loss = loss_outcome_prediction(args, train_ids, predicted_outcome, factual_outcome)
                    treatment_loss = loss_treatment_prediction(train_ids, predicted_treatment, factual_treatment)
                    loss = outcome_loss - args.beta*treatment_loss
                    t_loss = torch.sum(treatment_loss)
                    loss_log_t += t_loss.item()
                if args.disentangle == 0:
                    predicted_outcome = model(args, data)
                    loss = loss_outcome_prediction(args, train_ids, predicted_outcome, factual_outcome)
                loss = torch.sum(loss)
                loss_log += loss.item()  
                loss.backward()
                optimizer.step()
                time_iter = time.time()-start
            loss_log = loss_log/n_train
            loss_log_t = loss_log_t/n_train
            if epoch % 10 == 0:
                print('Epoch {}:'.format(epoch))

            val_loss, precision, recall, accuracy, auc = test_process(val)
            if args.disentangle == 1:
                val_mae_loss = test_treatment(val)
                mlflow.log_metric(key='val_mae_loss', value=val_mae_loss, step=n_step + t_step)

            if epoch % 10 == 0:
                print("FOLD {}, Time {:.4f} -- Validation loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}".format(
                    fold, time_iter, val_loss, precision, recall, accuracy, auc))
                if args.disentangle == 1:
                    print("FOLD {}, Time {:.4f} -- Treatment prediction training mae:{}".format(fold, time_iter, val_mae_loss))
            mlflow.log_metric(key='val_loss', value=val_loss, step=n_step)
            mlflow.log_metric(key='val_accuracy', value=accuracy, step=n_step)
            mlflow.log_metric(key='val_precision', value=precision, step=n_step)
            mlflow.log_metric(key='val_recall', value=recall, step=n_step)
            mlflow.log_metric(key='val_AUC', value=auc, step=n_step)

            if auc > max_auc:
                print("Epoch {}:  Model Saved !!!!!!!!!!".format(epoch))
                torch.save(model.state_dict(), 'model/model_predictor_{}_{}'.format(
                    args.exp_name, args.model_name_suffix))
                if args.disentangle == 1:
                    torch.save(classifier.state_dict(), 'model/model_classifier_{}_{}'.format(
                        args.exp_name_c, args.model_name_suffix))
                max_auc = auc
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break
            n_step += 1
        else:
            for batch_idx, train_ids in enumerate(train_loaders):
                train_ids_all += list(train_ids)
                n_train += len(train_ids)
                optimizer.zero_grad()
                if args.disentangle == 1:
                    embedding, predicted_outcome = model(args, data)
                    predicted_treatment = classifier(args, embedding)
                    outcome_loss = loss_outcome_prediction(args, train_ids, predicted_outcome, factual_outcome)
                    treatment_loss = loss_treatment_prediction(train_ids, predicted_treatment, factual_treatment)
                    loss = outcome_loss - args.beta*treatment_loss
                    t_loss = torch.sum(treatment_loss)
                    loss_log_t += t_loss.item()
                if args.disentangle == 0:
                    predicted_outcome = model(args, data)
                    loss = loss_outcome_prediction(args, train_ids, predicted_outcome, factual_outcome)
                loss = torch.sum(loss)
                loss_log += loss.item()  
                loss.backward()
                optimizer.step()
                time_iter = time.time()-start

            if epoch % 10 == 0:
                print('Epoch {}:'.format(epoch))

            val_loss, val_smape = test_process(val, 'test')
            if args.disentangle == 1:
                val_mae_loss = test_treatment(val)
                mlflow.log_metric(key='val_mae_loss', value=val_mae_loss, step=n_step + t_step)

            if epoch % 10 == 0:
                print("FOLD {}, Time {:.4f} -- Validation loss:{}, smape:{}".format(
                    fold, time_iter, val_loss, val_smape))
                if args.disentangle == 1:
                    print("FOLD {}, Time {:.4f} -- Validation treatment prediction mae:{}".format(fold, time_iter,
                                                                                                val_mae_loss))
            mlflow.log_metric(key='val_loss', value=val_loss, step=n_step)
            mlflow.log_metric(key='val_smape', value=val_smape, step=n_step)

            if val_loss < min_loss:
                best_epoch = epoch
                if epoch % 10 == 0:
                    print("!!!!!!!!!! Model Saved !!!!!!!!!!")
                torch.save(model.state_dict(), 'model/model_predictor_{}_{}'.format(
                    args.exp_name, args.model_name_suffix))
                if args.disentangle == 1:
                    torch.save(classifier.state_dict(), 'model/model_classifier_{}_{}'.format(
                        args.exp_name_c, args.model_name_suffix))
                min_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break
            n_step += 1

        # adversarial
        if epoch % 5 == 0 and epoch >0 and args.disentangle == 1:
            for epoch_adversarial in range(args.adversarial_epoch):  
                loss_log_a = 0.
                train_ids_all_a = []
                n_train_a = 0
                for batch_idx, train_ids in enumerate(train_loaders):
                    train_ids_all_a += list(train_ids)
                    n_train_a += len(train_ids)
                    optimizer_c.zero_grad()
                    embedding, _ = model(args, data)
                    predicted_treatment = classifier(args, embedding)
                    treatment_loss = loss_treatment_prediction(train_ids, predicted_treatment, factual_treatment)
                    loss = treatment_loss
                    loss = torch.sum(loss)
                    loss_log_a += loss.item()
                    loss.backward()
                    optimizer_c.step()
                    # 要加一个对classifier的评定
                mae_loss = test_treatment(data)
                mlflow.log_metric('adversarial_loss', value=mae_loss, step=t_step)
                t_step += 1
            val_mae_loss = test_treatment(val)
            if epoch % 10 == 0:
                print("FOLD {}, Time {:.4f} -- Adversarial Treatment prediction training mae:{}".format(fold, time_iter, val_mae_loss))
            mlflow.log_metric(key='val_mae_loss', value=val_mae_loss, step=n_step+t_step)
    mlflow.log_metric(key='best_epoch', value=best_epoch)


def test_process(data, state='train'):
    model.eval()
    data = data.to(args.device)

    if args.disentangle == 1:
        _, predicted_outcome = model(args, data)
    else:
        predicted_outcome = model(args, data)
    loss = loss_outcome_prediction(args, range(len(data.t)), predicted_outcome, data.outcome)
    loss = torch.mean(loss).item()

    def judge(x):
        if x >= args.yuzhi:
            return 1
        else:
            return 0

    if args.classify == 1:
        prediction_ori = predicted_outcome.cpu().detach().numpy()
        prediction = [judge(m) for m in prediction_ori]
        if args.use_yuzhi != 1:
            auc = roc_auc_score(data.outcome.cpu().detach().numpy(), prediction_ori)
        else:
            auc = roc_auc_score(data.outcome.cpu().detach().numpy(), prediction)
        precision = precision_score(data.outcome.cpu().detach().numpy(), prediction)
        recall = recall_score(data.outcome.cpu().detach().numpy(), prediction)
        accuracy = accuracy_score(data.outcome.cpu().detach().numpy(), prediction)
        return loss, precision, recall, accuracy, auc
    else:
        if state == 'train':
            mse = torch.nn.MSELoss(reduction='none')(predicted_outcome, data.outcome)
            mse = torch.mean(mse).item()
            return loss, mse
        if state == 'test': # smape
            smape = mean_absolute_percentage_error(data.outcome, predicted_outcome)
            return loss, smape


def test_treatment(data, state='Train'):
    classifier.eval()
    data = data.to(args.device)

    factual_treatment = data.t.view(-1, 1)
    if state=='Pretrain':
        predicted_treatment = classifier(args, data.x)
    else:
        embedding, _ = model(args, data)
        predicted_treatment = classifier(args, embedding)
    loss = loss_treatment_prediction(range(len(data.t)), predicted_treatment, factual_treatment)
    loss = torch.mean(loss).item()
    return loss


if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(
        description='Causal prediction network')
    parser.add_argument('--exp_name', type=str, default='test', help='exp_name')
    parser.add_argument('--exp_name_c', type=str, default='test_classifier', help='exp_name')
    parser.add_argument('-dp', '--dataset_path', type=str, default='Beibei_normed_ood1.npy',
                        help='node feature matrix data path')
    parser.add_argument('-dir', '--data_directory', type=str, default='data',
                        help='data_directory')
    parser.add_argument('-sd', '--seed', type=int,
                        default=666, help='random seed')
    parser.add_argument('-lr', '--lr', type=float,
                        default=0.0002, help='learning rate')
    parser.add_argument('-lra', '--lra', type=float,
                        default=0.0002, help='learning rate for the discriminator training')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=125, help='batch size')
    parser.add_argument('-wd', '--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('-e', '--epochs', type=int,
                        default=400, help='number of epochs')
    parser.add_argument('-dvs', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str,
                        default='main', help='model')
    parser.add_argument('-ba', '--base', type=str, default='mlp', help='base_model')
    parser.add_argument('-d', '--disentangle', type=int, default=1, help='whether to use the GCIL module')
    parser.add_argument('-n_folds', type=int, default=5, help='n_folds')
    parser.add_argument('-beta', type=float, default=0.2, help='balance coefficient for treament loss and outcome loss')
    parser.add_argument('-p', '--patience', type=int,
                        default=150, help='Patience')
    parser.add_argument('-pe', '--pretrain_epochs', type=int,
                        default=20, help='number of pretrain epochs')
    parser.add_argument('-yz', '--yuzhi', type=float,
                        default=0.5, help='line for judgement')
    parser.add_argument('-y', '--use_yuzhi', type=int, default=1, help='whether to round the result')
    parser.add_argument('-ae', '--adversarial_epoch', type=int, default=5, help='num of adversarial epochs')
    parser.add_argument('-c', '--classify', type=int, default=1, help='whether to classify or regress')
    parser.add_argument('-dr', '--dropout', type=float, default=0, help='whether to drop out parameters')
    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True  
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed_all(args.seed)  
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False   
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    start_time = time.time()

    print('------------------------- Loading data -------------------------')
    train_matrix, test_matrix = np.load(args.data_directory + '/' + args.dataset_path, allow_pickle=True)
    test = create_dataset(test_matrix, args)
    args.n_features = np.shape(train_matrix)[1] - 2
    args.n_embedding = args.n_features
    args.model_name_suffix = ''.join(random.sample(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e'], 8))

    train_ids, val_ids = split_train_val_test(
        np.shape(train_matrix)[0], args.n_folds, args.seed)

    mlflow.set_tracking_uri('beibei/mlflow')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(args.exp_name)
    except:
        experiments = client.get_experiment_by_name(args.exp_name)
        EXP_ID = experiments.experiment_id

    with mlflow.start_run(experiment_id=EXP_ID):
        # archive_path = mlflow.get_artifact_uri()
        mlflow.log_params(vars(args))

        test_loss, test_precision, test_recall, test_accuracy, test_auc, test_mae_loss = [], [], [], [], [], []
        test_mae, test_smape, test_a_mae = [], [], []
        val_f_mae, val_f_smape = [], []
        outcome = []
        trueoutcome = []
        for fold in range(args.n_folds):
            train = create_dataset(train_matrix[train_ids[fold], :], args)
            val = create_dataset(train_matrix[val_ids[fold], :], args)

            train_loaders, num_train = make_batch(
                train.x.size(0), args.batch_size, args.seed)

            print('\nFOLD {}, train {}, valid {}, test {}'.format(
                fold, num_train, np.shape(val.x)[0], np.shape(test.x)[0]))

            if args.disentangle == 1:       # adopting GCIL framework
                print('\n------------- Initialize Model -------------')
                model, optimizer, classifier, optimizer_c = init_model(args)

                if args.disentangle == 1:
                    print('\n------------- Pretraining -------------')
                    pretrain(args, train, train_loaders, val)

                print('\n------------- Training -------------')
                train_process(args, train, train_loaders, val)

                print('\n------------- loading -------------')
                model.load_state_dict(
                    torch.load('model/model_predictor_{}_{}'.format(
                        args.exp_name, args.model_name_suffix)))
                if args.classify == 1:  # classification problems
                    loss, precision, recall, accuracy, auc = test_process(
                        test)
                    if args.disentangle == 1:
                        classifier.load_state_dict(
                            torch.load('model/model_classifier_{}_{}'.format(
                                args.exp_name_c, args.model_name_suffix)))
                        mae_loss = test_treatment(test)
                        test_mae_loss.append(mae_loss)
                    test_loss.append(loss)
                    test_precision.append(precision)
                    test_recall.append(recall)
                    test_accuracy.append(accuracy)
                    test_auc.append(auc)

                    print('---------------------------------------')
                    print("Test loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}".format(
                        loss, precision, recall, accuracy, auc))
                    if args.disentangle == 1:
                        print("Treatment test loss:{}".format(mae_loss))
                else:   # regression problems
                    loss, smape = test_process(test, state='test')
                    test_mae.append(loss)
                    test_smape.append(smape)
                    if args.disentangle == 1:
                        classifier.load_state_dict(
                            torch.load('model/model_classifier_{}_{}'.format(
                                args.exp_name_c, args.model_name_suffix)))
                        mae_loss = test_treatment(test)
                        test_a_mae.append(mae_loss)
                    print('---------------------------------------')
                    print("Test mae loss:{}, smape:{}".format(
                        loss, smape))
                    if args.disentangle == 1:
                        print("Treatment test loss:{}".format(mae_loss))

            else:   # not using GCIL framework
                print('\n------------- Initialize Model -------------')
                model, optimizer = init_model(args)

                print('\n------------- Training -------------')
                train_process(args, train, train_loaders, val)  
                
                print('\n------------- loading -------------')
                model.load_state_dict(
                    torch.load('model/model_predictor_{}_{}'.format(
                        args.exp_name, args.model_name_suffix)))
                if args.classify == 1:
                    loss, precision, recall, accuracy, auc = test_process(
                        test)
                    test_loss.append(loss)
                    test_precision.append(precision)
                    test_recall.append(recall)
                    test_accuracy.append(accuracy)
                    test_auc.append(auc)

                    print('---------------------------------------')
                    print("Test loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}".format(
                        loss, precision, recall, accuracy, auc))
                else:
                    loss, smape = test_process(test, state='test')    
                    test_mae.append(loss)
                    test_smape.append(smape)
                    print('---------------------------------------')
                    print("Test mae loss:{}, smape:{}".format(
                        loss, smape))

        print('Total train time: {}', time.time() - start_time)
        if args.classify == 1:
            print('{}-fold cross validation avg loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}'.format(
                args.n_folds, np.mean(test_loss), np.mean(test_precision),
                np.mean(test_recall), np.mean(test_accuracy), np.mean(test_auc)))
            if args.disentangle == 1:
                print('{}-fold cross validation avg treatment prediction mae_loss:{}'.format(args.n_folds, np.mean(test_mae_loss)))
                mlflow.log_metric(key='test_a_mae_loss', value=np.mean(test_mae_loss))
            test_accuracy_trunc = [int(u * 10000) / 10000 for u in test_accuracy]
            test_auc_trunc = [int(u * 10000) / 10000 for u in test_auc]
            test_precision_trunc = [int(u * 10000) / 10000 for u in test_precision]
            test_recall_trunc = [int(u * 10000) / 10000 for u in test_recall]
            test_loss_trunc = [int(u * 10000) / 10000 for u in test_loss]
            mlflow.log_param(key="test_auc_all", value=str(test_auc_trunc))
            mlflow.log_param(key="test_acc_all", value=str(test_accuracy_trunc))
            mlflow.log_param(key="test_precision_all", value=str(test_precision_trunc))
            mlflow.log_param(key="test_recall_all", value=str(test_recall_trunc))
            mlflow.log_param(key="test_loss_all", value=str(test_loss_trunc))
            mlflow.log_metric(key='test_loss', value=np.mean(test_loss))
            mlflow.log_metric(key='test_accuracy', value=np.mean(test_accuracy))
            mlflow.log_metric(key='test_precision', value=np.mean(test_precision))
            mlflow.log_metric(key='test_recall', value=np.mean(test_recall))
            mlflow.log_metric(key='test_AUC', value=np.mean(test_auc))
        else:
            print('{}-fold cross validation avg mae loss:{}, smape:{}'.format(
                args.n_folds, np.mean(test_mae), np.mean(test_smape)))
            if args.disentangle == 1:
                print('{}-fold cross validation avg treatment prediction mae_loss:{}'.format(args.n_folds,
                                                                                             np.mean(test_a_mae)))
                mlflow.log_metric(key='test_a_mae_loss', value=np.mean(test_a_mae))
            test_smape_trunc = [int(u * 1000000000000000000) / 1000000000000000000 for u in test_smape]
            test_mae_trunc = [int(u * 1000000000) / 1000000000 for u in test_mae]
            mlflow.log_param(key="test_smape_all", value=str(test_smape_trunc))
            mlflow.log_param(key="test_mae_all", value=str(test_mae_trunc))
            mlflow.log_metric(key='test_smape', value=np.mean(test_smape))
            mlflow.log_metric(key='test_mae', value=np.mean(test_mae))
            mlflow.log_metric(key='test_mae_std', value=np.std(test_mae))
            mlflow.log_metric(key='test_smape_std', value=np.std(test_smape))







