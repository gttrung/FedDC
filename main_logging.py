
import os
import copy
import numpy as np
import random
import torch
import pandas as pd

from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import torch.nn as nn

from util.options import args_parser
from util.dynamic import separate_users, merge_users 
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.util import add_noise, lid_term, get_output
from util.dataset import get_dataset
from model.build_model import build_model

np.seterr(invalid='ignore')
np.set_printoptions(threshold=np.inf)
"""
Major framework of noise FL
"""

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # ---------------------------Separate new clients---------------------------

    dict_users, new_users = separate_users(args, dict_users)

    # ---------------------------Add Noise---------------------------

    y_train = np.array(dataset_train.targets)
    y_train_clean = y_train.copy()
    y_train_noisy, gamma_s = add_noise(args, y_train, dict_users, new_users)

    args.num_users = args.num_users - args.num_new_users
    args.frac1 = 1/args.num_users
    num_user_old = num_user_new = args.num_users
    dataset_train.targets = y_train_noisy

    settings = f'{args.dataset}_{args.method}_{args.num_new_users}_{args.level_n_new_system}'
    rootpath = f'./results/{settings}/'
    if not os.path.exists(rootpath):
          os.makedirs(rootpath)
    txtpath = rootpath + '%s_NEW_%d_%s_%s_NL_%.1f_NNL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_LRM_%.4f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d'\
        %(args.method, args.num_new_users, args.dataset, args.model, args.level_n_system, args.level_n_new_system, args.level_n_lowerb, args.iteration1, args.rounds1,
        args.rounds2, args.local_ep, args.frac1, args.frac2, args.lr, args.lr_min, args.relabel_ratio,
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

    f_log = open(txtpath + '_acc.txt', 'a')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(True)

    # build model
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')

    # client_base = np.random.randint(0, args.num_users, size=2).astype(int)
    # client_new = np.random.randint(args.num_users, args.num_users + args.num_new_users, size=2).astype(int)
    client_base = np.array([30,49])
    client_new = np.array([83,90])
    print(client_base, client_new)

    LID_accumulative_client = np.zeros(args.num_users)

    best_acc = 0.0
    acc_list = []
    loss_list = []

    if args.dataset == 'cifar10':
        predict_vector = ['sample index', 'true label', 'predicted label', 'airplane', 'automobile', 'bird', 'bat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
        predict_vector = ['sample index', 'true label', 'predicted label','apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # ------------------------------- first stage training -------------------------------

    for iteration in range(args.iteration1):
          
          path = "iteration_%d"%(iteration)
          if not os.path.exists(rootpath+ 'data_features/'+path+'/'):
            os.makedirs(rootpath+'data_features/'+path+'/')
          if not os.path.exists(rootpath+ 'data_features/'+path+'/t_sne/'):
            os.makedirs(rootpath+ 'data_features/'+path+'/t_sne/')

          if iteration == args.joining_round[0]:

            new_clients = np.arange(args.num_users, args.num_users + args.num_new_users*args.stage_ratio).astype(int)
            dict_users = merge_users(dict_users, new_users, args, stage = 1)
            args.num_users += len(new_clients)
            num_user_new = args.num_users
            args.frac1 = 1/args.num_users
            new_init_lid = 0
            while len(LID_accumulative_client) < args.num_users:
                  LID_accumulative_client = np.append(LID_accumulative_client, new_init_lid)

          LID_whole = np.zeros(len(y_train_noisy))
          loss_whole = np.zeros(len(y_train_noisy))
          LID_client = np.zeros(args.num_users)
          loss_accumulative_whole = np.zeros(len(y_train_noisy))

          # ----------------------Broadcast global model----------------------

          if iteration == 0:
              mu_list = np.zeros(args.num_users)
              if args.method == 'loss_thresh':
                 loss_thresh = 0
          else:
              mu_list = estimated_noisy_level

          if num_user_old < num_user_new:
              if args.method == 'loss_thresh':
                while len(mu_list) < args.num_users :
                    mu_list = np.append(mu_list,0.8)

              else:
                while len(mu_list) < args.num_users :
                    mu_list = np.append(mu_list,0)

          prob = [1 / args.num_users] * args.num_users 
          for _ in range(int(round(1/args.frac1, 0))):
              idxs_users = np.random.choice(range(args.num_users), int(np.round(args.num_users*args.frac1, 0)), p=prob)
              w_locals = []
              for idx in idxs_users:
                    prob[idx] = 0
                    if sum(prob) > 0:
                        prob = [prob[i] / sum(prob) for i in range(len(prob))]

                    net_local.load_state_dict(netglob.state_dict())
                    sample_idx = np.array(list(dict_users[idx]))
                    dataset_client = Subset(dataset_train, sample_idx)
                    loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                    # proximal term operation
                    mu_i = mu_list[idx]
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                    w, loss_local = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                                    w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)

                    net_local.load_state_dict(copy.deepcopy(w))
                    w_locals.append(copy.deepcopy(w))

                    acc = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                    acc_list.append(acc)
                    loss_list.append(loss_local)

                    if best_acc < acc:
                        best_acc = acc

                    local_output, loss_per_sample= get_output(loader, net_local.to(args.device), args, False, criterion)

                    if args.method == 'loss_thresh':
                        if iteration == 0:
                          loss_thresh += loss_local
                        if num_user_old < num_user_new:
                          if idx in new_clients:
                            
                            num_user_old += 1
                            if loss_local >= loss_thresh:
				    print(f'client {idx}: noisy')
				    LID_accumulative_client[idx] = np.mean(LID_accumulative_old)
                            else:
				    print(f'client {idx}: clean')
				    LID_accumulative_client[idx] = np.min(LID_accumulative_old)

                    f_log.write("iteration %d, epoch %d, client %d, loss: %.4f, acc: %.4f ,best_acc: %.4f\n" % (iteration, _, idx, loss_local, acc, best_acc))
                    f_log.flush()

                    if idx in client_base or idx in client_new:
                        feature_vector = local_output.copy()
                        feature_vector = np.insert(feature_vector, 0, np.argmax(local_output,axis=1), axis=1)
                        feature_vector = np.insert(feature_vector, 0, y_train[sample_idx], axis=1)
                        feature_vector = np.insert(feature_vector, 0, sample_idx, axis=1)
                        df = pd.DataFrame(data = feature_vector,
                                          columns = predict_vector)

                        df.to_csv(rootpath+'data_features/'+path+'/'+str(idx)+'.csv')
                        df = pd.read_csv(rootpath+'data_features/'+path+'/'+str(idx)+'.csv').values
                        features = df[:,4:]
                        samples_idx = df[:,1].astype(int)

                        with open(rootpath+ 'data_features/'+path+'/t_sne/'+str(idx)+'_labels.tsv','w') as f1:
                            for label in y_train[samples_idx]:
                                f1.write(str(label) + '\n')

                        with open(rootpath+ 'data_features/'+path+'/t_sne/'+str(idx)+'_features.tsv','w') as f2:
                            for fs in features:
                                for f_i in fs:
                                    f2.write(str(f_i) + '\t')
                                f2.write('\n')

                    LID_local = list(lid_term(local_output, local_output))
                    LID_whole[sample_idx] = LID_local
                    loss_whole[sample_idx] = loss_per_sample
                    LID_client[idx] = np.mean(LID_local)

              dict_len = [len(dict_users[idx]) for idx in idxs_users]
              w_glob = FedAvg(w_locals, dict_len)

              netglob.load_state_dict(copy.deepcopy(w_glob))

          if args.method == 'loss_thresh':
            if iteration == 0:
              loss_thresh /= args.num_users

          LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
          loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)
          LID_accumulative_old = LID_accumulative_client.copy()

          #----------------------Apply Gaussian Mixture Model to LID----------------------

          gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
              np.array(LID_accumulative_client).reshape(-1, 1))
          labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
          clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

          noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
          clean_set = np.where(labels_LID_accumulative == clean_label)[0]

          estimated_noisy_level = np.zeros(args.num_users)

          for client_id in noisy_set:
              
              sample_idx = np.array(list(dict_users[client_id]))
              loss = np.array(loss_accumulative_whole[sample_idx])
              gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
              labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
              gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

              pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
              estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
              y_train_noisy_new = np.array(dataset_train.targets)

          if args.correction:
            for idx in noisy_set:
                  
                  sample_idx = np.array(list(dict_users[idx]))
                  dataset_client = Subset(dataset_train, sample_idx)
                  loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                  loss = np.array(loss_accumulative_whole[sample_idx])
                  local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                  relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                  relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))

                  y_train_noisy_new = np.array(dataset_train.targets)
                  y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                  dataset_train.targets = y_train_noisy_new
    d_loss = {'loss': loss_list}
    df_loss = pd.DataFrame(d_loss)
    df_loss.to_csv(rootpath + 'loss_s1.txt')

    d_acc = {'acc': acc_list}
    df_acc = pd.DataFrame(d_acc)
    df_acc.to_csv(rootpath + 'acc_s1.txt')

    # reset the beta
    args.beta = 0
    # ------------------------------- second stage training -------------------------------
    if args.fine_tuning:

        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]
        prob = np.zeros(args.num_users)
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, len(selected_clean_idx))
        netglob = copy.deepcopy(netglob)
        acc_s2_list = []

        # add fl training
        for rnd in range(args.rounds1):

            w_locals, loss_locals = [], []

            if rnd == args.joining_round[1]:

              new_clients = np.arange(args.num_users, args.num_users + args.num_new_users*(1 - args.stage_ratio)).astype(int)
              dict_users = merge_users(dict_users, new_users, args, stage = 2)
              args.num_users += len(new_clients)
              num_user_new = args.num_users

              prob = np.append(prob, np.zeros(int(args.num_new_users*(1-args.stage_ratio))))
              m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
              m = min(m, len(selected_clean_idx))
              prob[prob!=0] = 1 / (len(np.where(prob!=0)[0])  + len(new_clients))
              prob[new_clients] = 1 / (len(np.where(prob!=0)[0])  + len(new_clients))

            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

            for idx in idxs_users:  # training over the subset

                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                if args.method == 'loss_thresh':
			if num_user_old < num_user_new and idx in new_clients:
				num_user_old += 1
				args.beta = 5
				w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
									   w_g=netglob.to(args.device), epoch=args.local_ep, mu=0.8)
				# reset beta for other clients
				args.beta = 0
				if loss_local >= loss_thresh:
					print(f'client {idx}: noisy')
					noisy_set = np.append(noisy_set, idx)
					prob[prob!=0] = 1 / (len(np.where(prob!=0)[0]) - 1)
					prob[idx] = 0
					idxs_users = np.setdiff1d(idxs_users, idx)
					continue
				else:
					w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
										   w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
		else:
			w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
								   w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
                w_locals.append(copy.deepcopy(w_local))  # store updated model
                loss_locals.append(copy.deepcopy(loss_local))
                loss_list.append(loss_local)
            
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            acc_s2  = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
            acc_s2_list.append(acc_s2)
            if best_acc < acc_s2:
                best_acc = acc_s2
            f_log.write("fine tuning stage round %d, test acc: %.4f, best acc: %.4f \n" % (rnd, acc_s2, best_acc))
            f_log.flush()

        if args.correction:

            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                y_train_noisy_new = np.array(dataset_train.targets)

                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                dataset_train.targets = y_train_noisy_new
                
        d_loss = {'loss': loss_list}
        df_loss = pd.DataFrame(d_loss)
        df_loss.to_csv(rootpath + 'loss_s2.txt')
    
        d_acc = {'acc': acc_s2_list}
        df_acc = pd.DataFrame(d_acc)
        df_acc.to_csv(rootpath + 'acc_s2.txt')
    

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
        f_log.write("third stage round %d, test acc: %.4f, best acc: %.4f \n" % (rnd, acc_s3, best_acc))
        f_log.flush()

    d_loss = {'loss': loss_list}
    df_loss = pd.DataFrame(d_loss)
    df_loss.to_csv(rootpath + 'loss_s3.txt')

    d_acc = {'acc': acc_s3_list}
    df_acc = pd.DataFrame(d_acc)
    df_acc.to_csv(rootpath + 'acc_s3.txt')

    torch.cuda.empty_cache()
