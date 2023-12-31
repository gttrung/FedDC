# FedDC

## 1. Parameters

**1.1. Descriptions**

In addition to the FedCorr parameters that are used in FedDC, we add some new parameters to FedDC.

| new parameters       | description                                                                                    |
|----------------------|------------------------------------------------------------------------------------------------|
| `lr_min`             | minimum learning rate for lr cyclic                                                            |
| `level_n_new_system` | fraction of new noisy clients ($\sigma$ in paper)                                              |
| `num_new_users`      | number of new clients clients                                                                  |
| `stage_ratio`        | ratio of new clients per stage in stage 1 and 2                                                |
| `joining_round`      | iteration/round of joining new clients in stage 1 and 2                                        |
| `lr_cyclic`          | whether to use cyclic learning rate                                                            |
| `dynamic`            | whether to include dynamic clients                                                             |
| `method`             | methods to detect new noisy client: losh_thresh for our method, no_method for original FedCorr |

**1.2. Values**

For the old FedCorr parameters we set them largely the same as the FedCorr experiments on CIFAR10 and CIFAR100. The only old parameter changed is `local_ep` and the new parameters with their values are listed below.

| parameters           | CIFAR10 | CIFAR100 |
|----------------------|---------|----------|
| `local_ep`           | 6       | 6        |
| `lr_min`             | 0.001   | 0.001    |
| `num_new_users`      | 10      | 5        |
| `joining_round`      | 3, 498  | 8, 448   |
| `stage_ratio`        | 0.5     | 0.5      |

## 2. Quick start
+ To train on CIFAR-10 with IID data partition and noise setting $(\rho,\tau,\sigma)=(0.6,0.5,0.6)$, over 90 clients and 10 new clients with 0.5 `stage_ratio`:

```
python main.py --dataset cifar10 --model resnet18 --method loss_thresh --level_n_system 0.6 --level_n_lowerb 0.5 --level_n_new_system 0.6 --iteration1 5 --rounds1 500 --rounds2 450 --seed 1 --mixup --lr 0.03 --lr_min 0.001 --beta 5 --num_users 100 --num_new_users 10 --joining_round 3 498 --stage_ratio 0.5

```
+ To train on CIFAR-10 with Non-IID data partition and noise setting $(\rho,\tau,\sigma)=(0.6,0.5,0.6)$, over 90 clients and 10 new clients with 0.5 `stage_ratio`:

```
python main.py --dataset cifar10 --iid --model resnet18 --method loss_thresh --level_n_system 0.6 --level_n_lowerb 0.5 --level_n_new_system 0.6 --iteration1 5 --rounds1 500 --rounds2 450 --seed 1 --mixup --lr 0.03 --lr_min 0.001 --beta 5 --num_users 100 --num_new_users 10 --joining_round 3 498 --stage_ratio 0.5

```

+ To train on CIFAR-100 with IID data partition and noise setting $(\rho,\tau,\sigma)=(0.6,0.5,0.6)$, over 45 clients and 5 new clients with 0.5 `stage_ratio`:

```
python main.py --dataset cifar100 --model resnet34 --method loss_thresh --level_n_system 0.6 --level_n_lowerb 0.5 --level_n_new_system 0.6 --iteration1 10 --rounds1 150 --rounds2 450 --seed 1 --mixup --lr 0.01 --lr_min 0.001 --beta 5 --num_users 50 --num_new_users 5 --joining_round 8 448 --stage_ratio 0.5
```
+ To train on CIFAR-100 with Non-IID data partition and noise setting $(\rho,\tau,\sigma)=(0.6,0.5,0.6)$, over 45 clients and 5 new clients with 0.5 `stage_ratio`:

```
python main.py --dataset cifar100 --iid --model resnet34 --method loss_thresh --level_n_system 0.6 --level_n_lowerb 0.5 --level_n_new_system 0.6 --iteration1 10 --rounds1 150 --rounds2 450 --seed 1 --mixup --lr 0.01 --lr_min 0.001 --beta 5 --num_users 50 --num_new_users 5 --joining_round 8 448 --stage_ratio 0.5
```
+ To train with original FedCorr, simply change the `method` argument to **no_method** instead of **loss_thresh**.
