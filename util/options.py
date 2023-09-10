# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=500, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=450, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=6, help="number of local epochs, 6 for loss_thresh, 5 for no_method")
    parser.add_argument('--frac1', type=float, default=0.01,help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.1, help="fration of selected clients in fine-tuning and usual training stage")

    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--lr_min', type=float, default=0.001, help="minimum learning rate for cyclic lr")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")
    parser.add_argument('--beta', type=float, default=5, help="coefficient for local proximal, 0 for fedavg, 1 for fedprox, 5 for noise fl")

    # noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--level_n_system', type=float, default=0.6, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.4, help="lower bound of noise level")
    parser.add_argument('--level_n_new_system', type=float, default=0.6, help="fraction of new noisy clients")

    # dynamic clients arguments
    parser.add_argument('--num_new_users', type=int, default=10, help="number of new clients clients")
    parser.add_argument('--stage_ratio', type=float, default=0.5, help="ratio of new clients per stage in stage 1 and 2")
    parser.add_argument('--joining_round', nargs='+', type=int, default=[3, 498], help="iteration/round of joining new clients in stage 1 and 2")

    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")

    # ablation study
    parser.add_argument('--lr_cyclic', action='store_false', help="whether to use cyclic learning rate")
    parser.add_argument('--dynamic', action='store_false', help="whether to include dynamic clients")
    parser.add_argument('--fine_tuning', action='store_false', help="whether to include fine-tuning stage")
    parser.add_argument('--correction', action='store_false', help="whether to correct noisy labels")
    parser.add_argument('--method', type=str, default='loss_thresh', help="methods to detect new noisy client: losh_thresh for our method, no_method for original FedCorr")

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_false', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=1, help="random seed, default: 1")
    parser.add_argument('--mixup', action='store_false')
    parser.add_argument('--alpha', type=float, default=1, help="0.1,1,5")

    args = parser.parse_args(args=[])

    return parser.parse_args()