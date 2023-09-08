import sys
import os
import copy
import numpy as np
import random
import torch

if "ipykernel" in sys.modules:
    import tqdm.notebook as tqdm
else:
    from tqdm import tqdm

from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import torch.nn as nn

from util.options import args_parser
from util.dynamic import merge_users, separate_users
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

    rootpath = './result/'
    if not os.path.exists(rootpath):
          os.makedirs(rootpath)
    txtpath = rootpath + '%s_%s_NL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d'\
        %(args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.iteration1, args.rounds1,
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
    torch.set_grad_enabled(True)

    # build model
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')

    LID_accumulative_client = np.zeros(args.num_users)

    best_acc = 0.0
    
    # ------------------------------- first stage training -------------------------------

    for iteration in tqdm(range(args.iteration1)):

          if iteration == args.joining_round[0]:

            new_client = np.arange(args.num_users, args.num_users + args.num_new_users*args.stage_ratio ).astype(int)
            dict_users = merge_users(dict_users, new_users)
            args.num_users += int(args.num_new_users*args.stage_ratio )
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
          for _ in tqdm(range(int(round(1/args.frac1, 0)))):
              idxs_users = np.random.choice(range(args.num_users), int(np.round(args.num_users*args.frac1,0)), p=prob)
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
                    w, loss_train = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                                    w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)

                    net_local.load_state_dict(copy.deepcopy(w))
                    w_locals.append(copy.deepcopy(w))

                    acc = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                    if best_acc < acc:
                        best_acc = acc

                    local_output, loss_per_sample= get_output(loader, net_local.to(args.device), args, False, criterion)

                    if args.method == 'loss_thresh':
                        if iteration == 0:
                          loss_thresh += loss_train
                        if iteration == args.joining_round[0]:
                          if idx in new_client:
                            if loss_train >= loss_thresh:
                                LID_accumulative_client[idx] = np.mean(LID_accumulative_old)
                            else:
                              LID_accumulative_client[idx] = np.min(LID_accumulative_old)

                    f_acc.write("iteration %d, epoch %d, client %d, loss: %.4f, acc: %.4f ,best_acc: %.4f\n" % (iteration, _, idx, loss_train, acc, best_acc))
                    f_acc.flush()

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

          num_user_old = args.num_users

    # reset the beta
    args.beta = 0
    torch.cuda.empty_cache()