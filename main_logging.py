
# python version 3.7.1
# -*- coding: utf-8 -*-

import os

try:
  import umap.umap_ as umap
except ImportError:
  print("Trying to install required module: umap\n")
  os.system('python -m pip install umap umap-learn')
import umap.umap_ as umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
import random
import torch
from torch.utils.data import Subset
# import umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.util import add_noise, get_output, kl_divergence_matrix, js_divergence
from util.dataset import get_dataset, ProxyDataset
from model.build_model import build_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

np.set_printoptions(threshold=np.inf)
"""
Major framework of noise FL
"""

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print('---------------------- Arguments ----------------------\n')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('\n-----------------------------------------------------\n')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    dataset_train, dataset_test, dict_users = get_dataset(args)
    # _, proxy_data, __, proxy_labels = train_test_split(dataset_test.data, dataset_test.targets, test_size=0.1, random_state=0, stratify=dataset_test.targets)
    dataset_proxy = ProxyDataset(dataset_test)
    # print(dataset_proxy.idxs)
    # samples_per_label = np.array([len(np.where(np.array(dataset_proxy.targets) == i)[0]) for i in range(args.num_classes)])
    # print(samples_per_label)
    new_users = {}
    # print(proxy_data.shape,dataset_train.data.shape)

    # ---------------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_test = np.array(dataset_test.targets)
    print('\n---------------------- Add Noise ----------------------\n')
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users,new_users)
    print('\n-----------------------------------------------------\n')
    dataset_train.targets = y_train_noisy

    settings = f'{args.noise_type}_{args.dataset}_iid_{args.num_users}_sys_{args.level_n_system}_low_{args.level_n_lowerb}' if args.iid \
        else f'{args.noise_type}_{args.dataset}_non_iid_p_{args.non_iid_prob_class}_dirich_{args.alpha_dirichlet}_{args.num_users}_sys_{args.level_n_system}_low_{args.level_n_lowerb}'
    rootpath = f'./results/{settings}/'

    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    # if not os.path.exists(rootpath + 'real_noise_level/'):
    #     os.makedirs(rootpath + 'real_noise_level/')
    # if not os.path.exists(rootpath + 'estimated_noise_level/'):
    #     os.makedirs(rootpath + 'estimated_noise_level/')
    # if not os.path.exists(rootpath + 'acc_correction/'):
    #     os.makedirs(rootpath + 'acc_correction/')
    dict_rnl={'rnl':real_noise_level}
    df_rnl = pd.DataFrame(dict_rnl)
    df_rnl.to_csv(rootpath + 'real_noise_level.txt')
    
    txtpath = rootpath + '%s_%s_%s_NL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d' % (
        args.noise_type, args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.iteration1, args.rounds1,
        args.rounds2, args.local_ep, args.frac1, args.frac2, args.lr, args.relabel_ratio,
        args.confidence_thres, args.clean_set_thres, args.beta, args.seed)

    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    if args.fine_tuning:
        txtpath += "_FT"
    if args.correction:
        txtpath += "_CORR"
    if args.mixup:
        txtpath += "_Mix_%.1f" % (args.alpha)

    f_acc = open(txtpath + '_acc.txt', 'a')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build model
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    sim_per_client = np.zeros((args.num_users, args.num_users))
    loss_accumulative_whole = np.zeros(len(y_train))
    sim_client = np.zeros(args.num_users)
    best_acc = 0.0
    # acc_correct_whole = []
    n_clusters = 2
    loss_client = np.zeros(args.num_users)
    normalized_acc = np.zeros(args.num_users)

    dict_pacc = {}
    dict_nacc = {}
    dict_mu = {}
    dict_correct_whole = {}
    
    for iteration in range(args.iteration1):

        path = "iteration_%d"%(iteration)
        # if iteration == 0:

        loss_whole = np.zeros(len(y_train))
        
        # acc_per_round = np.array([])
        output_whole = ([ [] for _ in range(args.num_users) ])
        acc_per_client = ([ [] for _ in range(args.num_users) ])
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level

        # ----------------------Broadcast global model----------------------

        prob = [1 / args.num_users] * args.num_users
        net_local_clients = ([ [] for _ in range(args.num_users) ])
        for _ in range(int(1/args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users*args.frac1), p=prob,replace=False)
            w_locals =  [] # ([ [] for _ in range(len(idxs_users)) ])
            for index, idx in enumerate(idxs_users):
                client = "client_%d"%(idx)
                if not os.path.exists(rootpath+ 'data_features/'+path+'/'+client+'/'):
                    os.makedirs(rootpath+'data_features/'+path+'/'+client+'/')
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                # test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=100, shuffle=False)
                proxy_loader = torch.utils.data.DataLoader(dataset=dataset_proxy, batch_size=100, shuffle=False)

                # proximal term operation
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                                w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_list[idx])

                net_local.load_state_dict(copy.deepcopy(w))
                # w_locals[index] = copy.deepcopy(w)
                net_local_clients[idx] = copy.deepcopy(w)
                w_locals.append(copy.deepcopy(w))
                acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                # normalized_acc[idx] = acc_t
                loss_client[idx] += loss
                # acc_per_round = np.append(acc_per_round,acc_t)
                if best_acc < acc_t:
                  best_acc = acc_t

                local_output, loss_per_sample = get_output(loader, net_local.to(args.device), args, False, criterion)
                # local_features, loss_per_sample = get_output(loader, net_local.to(args.device), args, True, criterion)

                proxy_output, loss_per_test = get_output(proxy_loader, net_local.to(args.device), args, False, criterion)
                # global_features, loss_per_test = get_output(proxy_loader, net_local.to(args.device), args, True, criterion)

                output_whole[idx] = proxy_output
                
                # ---------------------local output---------------------

                # with open(rootpath+'data_features/'+path+'/'+client+'/local_labels.tsv','w') as f1:
                #       for label in y_train[sample_idx]:
                #           f1.write(str(label) + '\n')

                # with open(rootpath+'data_features/'+path+'/'+client+'/local_output.tsv','w') as f2:
                #       for fs in local_output:
                #           for f_i in fs:
                #               f2.write(str(f_i) + ',')
                #           f2.write('\n')

                # with open(rootpath+'data_features/'+path+'/'+client+'/local_features.tsv','w') as f3:
                #       for fs in local_features:
                #           for f_i in fs:
                #               f3.write(str(f_i) + ',')
                #           f3.write('\n')

                # ---------------------global output---------------------

                with open(rootpath+'data_features/'+path+'/'+client+'/global_labels.tsv','w') as f4:
                      for label in np.array(dataset_proxy.targets):
                          f4.write(str(label) + '\n')

                with open(rootpath+'data_features/'+path+'/'+client+'/proxy_output.tsv','w') as f5:
                      for fs in proxy_output:
                          for i, f_i in enumerate(fs):
                                if i == len(fs)-1:
                                    f5.write(str(f_i))
                                else:
                                    f5.write(str(f_i) + ',')
                          f5.write('\n')

                # with open(rootpath+'data_features/'+path+'/'+client+'/global_features.tsv','w') as f6:
                #       for fs in global_features:
                #           for f_i in fs:
                #               f6.write(str(f_i) + ',')
                #           f6.write('\n')
                
                # f_acc.write("---------------------------------------------------------------------------\n")
                f_acc.write("iteration %d, round %d, client %d, acc: %.4f, best acc: %.4f \n" \
                          % (iteration, _, idx, acc_t, best_acc))
                f_acc.flush()

                loss_whole[sample_idx] = loss_per_sample
                # break
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len, mu_list[idxs_users])

            netglob.load_state_dict(copy.deepcopy(w_glob))
            # break
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)
        
         # ---------------------- Compute the similarity between clients ----------------------

        normalized_acc = np.zeros(args.num_users)
        for idx in range(args.num_users):
            net_local.load_state_dict(copy.deepcopy(net_local_clients[idx]))
            proxy_loader = torch.utils.data.DataLoader(dataset=dataset_proxy, batch_size=100, shuffle=False)
            proxy_output, _ = get_output(proxy_loader, net_local.to(args.device), args, False, criterion)
            predict_label = np.argmax(proxy_output, axis=1)
            acc_per_class = np.zeros(args.num_classes)
            for i in range(args.num_classes):
                acc_per_class[i] = np.where(predict_label[np.where(np.array(dataset_proxy.targets) == i)[0]] == i)[0].shape[0] / np.where(np.array(dataset_proxy.targets) == i)[0].shape[0]
            acc_per_client[idx] = acc_per_class
            acc_proxy = np.where(predict_label == np.array(dataset_proxy.targets))[0].shape[0] / len(predict_label)
            # acc_proxy = globaltest(copy.deepcopy(net_local).to(args.device), dataset_proxy, args)
            normalized_acc[idx] = acc_proxy
            
        # dict_pacc[f'iteration_{iteration}'] = acc_per_client
        # df_pacc = pd.DataFrame(dict_pacc)
        # df_pacc.to_csv(rootpath + f'acc_per_client_iteration_{iteration}.txt')
        dict_nacc[f'iteration_{iteration}'] = normalized_acc
        
        normalized_acc /= np.max(normalized_acc)
        
        for idx_a in range(args.num_users):
            for idx_b in range(idx_a+1,args.num_users):
                sim_per_samples = js_divergence(output_whole[idx_a], output_whole[idx_b]) \
                                    *(np.abs(normalized_acc[idx_a] - normalized_acc[idx_b]) + 1)
                # sim_per_samples = ( kl_divergence_matrix(output_whole[idx_a], output_whole[idx_b]) \
                #                     + kl_divergence_matrix(output_whole[idx_b], output_whole[idx_a]) ) \
                #                     *(np.abs(normalized_acc[idx_a]-normalized_acc[idx_b]) + 1)

                # sim_per_samples = kl_divergence_matrix(output_whole[idx_a], output_whole[idx_b])*normalized_acc[idx_a] \
                #                     + kl_divergence_matrix(output_whole[idx_b], output_whole[idx_a])*normalized_acc[idx_b] \

                sim_per_sample = np.mean(sim_per_samples)
                sim_per_client[idx_a][idx_b] += sim_per_sample
                sim_per_client[idx_b][idx_a] += sim_per_sample
                # sim_per_client[idx_a][idx_b] = sim_per_sample
                # sim_per_client[idx_b][idx_a] = sim_per_sample

        # ----------------------Clustering clients----------------------

        # Fit the KMeans model to the clusterable_embedding
        
        if n_clusters > 1:
            print('---------------------- Kmeans Clustering ----------------------')
            kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(sim_per_client)

            # Get the cluster labels
            cluster_labels = kmeans.labels_
            kmenas_clean_cluster = np.where(cluster_labels == 0)[0]
            kmenas_noisy_cluster = np.where(cluster_labels == 1)[0]
            # cluster_acc = np.array([np.mean(normalized_acc[cluster_labels == 0]),np.mean(normalized_acc[cluster_labels == 1])])
            # # cluster_loss = np.array([np.mean(loss_client[cluster_labels == 0]),np.mean(loss_client[cluster_labels == 1])])
            # clean_cluster = np.argmax(cluster_acc)
            # noisy_cluster = np.argmin(cluster_acc)
            # if clean_cluster != 0:
            #     print("Clean cluster is not 0 -> Swapping")
            #     cluster_labels = np.where(cluster_labels == clean_cluster, noisy_cluster, np.where(cluster_labels == noisy_cluster, clean_cluster, cluster_labels))

            for clean_client in kmenas_clean_cluster:
                sim_client[clean_client] += np.mean(sim_per_client[clean_client][kmenas_clean_cluster])
            for noisy_client in kmenas_noisy_cluster:
                sim_client[noisy_client] += np.mean(sim_per_client[noisy_client][kmenas_noisy_cluster])
            if np.mean(sim_client[kmenas_clean_cluster]) < np.mean(sim_client[kmenas_noisy_cluster]):
                
                print("Clean cluster is not 0 -> Swapping")
                cluster_labels = np.where(cluster_labels == 0, 1, np.where(cluster_labels == 1, 0, cluster_labels))
            clean_set = np.where(cluster_labels == 0)[0]
            noisy_set = np.where(cluster_labels == 1)[0]
            
            # visualize the clustering
            clusterable_embedding = umap.UMAP(
                n_neighbors=10,
                min_dist=0.0,
                n_components=2,
                random_state=args.seed,
            ).fit_transform(sim_per_client)
            
            centroids = kmeans.cluster_centers_
            labels_str = ['clean','noisy']
            #plotting the results:
            plt.figure()
            for i in range(n_clusters):
                plt.scatter(clusterable_embedding[gamma_s == i, 0] , clusterable_embedding[gamma_s == i , 1] , label = labels_str[i])
            # plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
            plt.legend()
            plt.savefig(rootpath + f'UMAP_iteration_{iteration}.png')
        
        # ---------------------- Compute the accuracy of kmeans ----------------------

        acc_kmeans = np.where(cluster_labels == gamma_s)[0].shape[0]/len(cluster_labels)
        print("iteration %d, acc of kmeans: %.4f \n" % (iteration, acc_kmeans))
        print('-----------------------------------------------------\n')
        
        # ---------------------- Swap stage ----------------------
        
        # ratio_set = clean_set.shape[0]/noisy_set.shape[0]
        # if ratio_set < args.ratio_set_thres:
        #     print("Clean set is larger than noisy set -> Stop using Kmeans")
        #     n_cluster = 1
        #     noisy_set = np.where(estimated_noisy_level > args.clean_set_thres)[0]
        #     clean_set = np.where(estimated_noisy_level <= args.clean_set_thres)[0]
        
        # TODO: Remove this if decentrailized FL work
        
        # ---------------------- Update with clean client ----------------------

        # dict_len = [len(dict_users[idx]) for idx in idxs_users]
        # clean_dict_len = [len(dict_users[idx]) for idx in clean_set]
        # w_clean_locals = [w_locals[idx] for idx in clean_set]
        # w_glob = FedAvg(w_clean_locals, clean_dict_len)
        # netglob.load_state_dict(copy.deepcopy(w_glob))

        # ----------------------Correction with noisy client----------------------

        estimated_noisy_level = np.zeros(args.num_users)
        real_noise_level_new = -np.ones(args.num_users)
        # acc_correction = np.zeros(args.num_users)

        for noisy_client in noisy_set:

            sample_idx = np.array(list(dict_users[noisy_client]))
            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            estimated_noisy_level[noisy_client] = len(pred_n) / len(sample_idx)
            # y_train_noisy_new = np.array(dataset_train.targets)
            
        dict_mu[f'iteration_{iteration}'] = estimated_noisy_level

        if args.correction:
            for idx in noisy_set:

                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                # dynamic_confidence_thres = 1 - 0.5 * acc_per_client[idx] / np.max(acc_per_client[idx])
                dynamic_confidence_thres = 1 - acc_per_client[idx] / np.max(acc_per_client[idx])
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                predicted_label = np.argmax(local_output, axis=1)
                relabel_idx = list(set(np.where(np.max(local_output, axis=1) > dynamic_confidence_thres[predicted_label])[0]) & set(relabel_idx))

                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                real_noise_level_new[idx] = np.where(y_train_noisy_new[sample_idx] != y_train[sample_idx])[0].shape[0] / len(sample_idx)
                # acc_correction[idx] = np.where(y_train_noisy_new[sample_idx] == y_train[sample_idx])[0].shape[0] / len(sample_idx)
                dataset_train.targets = y_train_noisy_new
                
            dict_rnl[f'iteration_{iteration}'] = real_noise_level_new
            dict_correct_whole[f'iteration_{iteration}'] = np.where(y_train_noisy_new == y_train)[0].shape[0] / len(y_train)
            
        df_nacc = pd.DataFrame(dict_nacc)
        df_nacc.to_csv(rootpath + f'normalized_acc.txt')
        
        df_mu = pd.DataFrame(dict_mu)
        df_mu.to_csv(rootpath + f'estimated_noisy_level.txt')
                
        df_rnl = pd.DataFrame(dict_rnl)
        df_rnl.to_csv(rootpath + f'real_noise_level.txt')
        # df_correct = pd.DataFrame({'acc_correction': acc_correction})
        # df_correct.to_csv(rootpath + f'acc_correction/acc_correction_stage_1_iteration_{iteration}.txt')
        # dict_correct_whole[f'iteration_{iteration}'] =  acc_correct_whole
        df_correct_whole = pd.DataFrame(dict_correct_whole, index = [0])
        df_correct_whole.to_csv(rootpath + f'acc_correct_whole.txt')

    # ------------------------------- second stage training -------------------------------
    
    args.local_bs = args.normal_local_bs
    if args.fine_tuning:
        # prob = np.zeros(args.num_users) # np.zeros(100)
        prob[clean_set] = 1 / len(clean_set)
        # prob = [1 / args.num_users] * args.num_users
        m = max(int(args.frac2 * len(clean_set)), 1)  # num_select_clients
        m = min(m, len(clean_set))
        netglob = copy.deepcopy(netglob)
        # add fl training
        real_noise_level_list = []
        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            for idx in idxs_users:  # training over the subset
                sample_idx = np.array(list(dict_users[idx]))
                # if idx in clean_set:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                            w_g=netglob.to(args.device), epoch=args.local_ep,  mu=0)
                # elif idx in noisy_set:
                #     if args.correction:
                        
                #         dataset_client = Subset(dataset_train, sample_idx)
                #         loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                #         proxy_loader = torch.utils.data.DataLoader(dataset=dataset_proxy, batch_size=100, shuffle=False)
                #         glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                #         proxy_output, __ = get_output(proxy_loader, netglob.to(args.device), args, False, criterion)
                        
                #         y_predicted = np.argmax(glob_output, axis=1)
                #         proxy_predicted = np.argmax(proxy_output, axis=1)
                #         acc_per_class = np.zeros(args.num_classes)
                #         for i in range(args.num_classes):
                #             acc_per_class[i] = np.where(proxy_predicted[np.where(np.array(dataset_proxy.targets) == i)[0]] == i)[0].shape[0] / np.where(np.array(dataset_proxy.targets) == i)[0].shape[0]
                #         dynamic_confidence_thres = 1 - acc_per_class / np.max(acc_per_class)
                        
                #         relabel_idx = np.where(np.max(glob_output, axis=1) > dynamic_confidence_thres[y_predicted])[0]
                #         y_train_noisy_new = np.array(dataset_train.targets)
                #         y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                #         real_noise_level_new[idx] = np.where(y_train_noisy_new[sample_idx] != y_train[sample_idx])[0].shape[0] / len(sample_idx)
                #         dataset_train.targets = y_train_noisy_new
                    
                #     local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                #     w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                #                                             w_g=netglob.to(args.device), epoch=args.local_ep,  mu=estimated_noisy_level[idx])
                    
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))
            
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len, mu_list[idxs_users])
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))
    
            acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
            if best_acc < acc_s2:
                best_acc = acc_s2
            f_acc.write("fine tuning with clean set round %d, clients: %s, test acc: %.4f, best acc: %.4f" % (rnd, idxs_users ,acc_s2, best_acc))
            f_acc.flush()

        if args.correction:
            real_noise_level_new = -np.ones(args.num_users)
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                dynamic_confidence_thres = 1 - acc_per_client[idx] / np.max(acc_per_client[idx])
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > dynamic_confidence_thres[y_predicted])[0]
                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                real_noise_level_new[idx] = np.where(y_train_noisy_new[sample_idx] != y_train[sample_idx])[0].shape[0] / len(sample_idx)
                dataset_train.targets = y_train_noisy_new
            
            dict_rnl[f'stage_2'] = real_noise_level_new
            dict_correct_whole[f'stage_2'] = np.where(y_train_noisy_new == y_train)[0].shape[0] / len(y_train)
                
            df_rnl = pd.DataFrame(dict_rnl)
            df_rnl.to_csv(rootpath + f'real_noise_level.txt')
            df_correct_whole = pd.DataFrame(dict_correct_whole, index = [0])
            df_correct_whole.to_csv(rootpath + f'acc_correct_whole.txt')
            
    # ---------------------------- third stage training -------------------------------
    
    # third stage hyper-parameter initialization
    acc_s3_list = []
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                        w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc_s3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        acc_s3_list.append(acc_s3)
        if best_acc < acc_s3:
            best_acc = acc_s3
        f_acc.write("third stage round %d - clients: %s, test acc: %.4f, best acc: %.4f \n" % (rnd, idxs_users, acc_s3, best_acc))
        f_acc.flush()
