# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
from matplotlib import pylab as plt
from os import path
from numpy import array, maximum, minimum, median
import numpy

# %%
# %matplotlib inline
plt.style.use(['dark_background', 'bmh'])
plt.rc('axes', facecolor='k')
plt.rc('figure', facecolor='k')
plt.rc('figure', figsize=(20,5))

# %%
N = 20
n = dict(); s = 'offroad'; d = 'dtr_orig'
n[s] = '/home/sk7685/ppuu_experiments/offroad_cost/models/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed={seed}-novaluestep{step}.model.log'
n[d] = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed={seed}-novaluestep{step}.model.log'
names = n
steps = [(i + 1) * 5000 for i in range(N)]

seeds = dict(
    offroad=[i+1 for i in range(3)],
    dtr_orig=[i + 1 for i in range(3)],
)
success = {k: list(list() for seed in seeds[k]) for k in seeds}

for k in seeds:
    for seed in seeds[k]:
        for step in steps:
            file_name = path.join(names[k].format(seed=seed, step=step))
            with open(file_name) as f:
                success[k][seed - 1].append(float(f.readlines()[-1].split()[-1]))

# %%
success_arr = {k: array(success[k]) for k in names}
# stats = ('min', 'max', 'median')
for k in names:
    plt.plot(
        array(steps) / 1e3, numpy.median(success_arr[k], 0),
        label=f'{k}',
        linewidth=2,
    )
for k in names:
    plt.fill_between(
        array(steps) / 1e3, success_arr[k].min(0), success_arr[k].max(0),
        alpha=.5,
    )
plt.grid(True)
plt.xlabel('steps [k–]')
plt.ylabel('success rate')
plt.legend(ncol=7)
plt.ylim([0.50, 0.85])
plt.xlim([5, 105])
plt.title('With and Without the Offroad Cost')
plt.xticks(range(10, 100 + 10, 10));
plt.show()
#plt.savefig('Rgr-vs-hrd-success_rate-min-max.png', bbox_inches = 'tight')

# %%
N = 20
n = dict(); s = 'softmax'; d = 'dtr_orig'
n[s] = '/home/sk7685/ppuu_experiments/softmax/models/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed={seed}-novaluestep{step}.model.log'
n[d] = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed={seed}-novaluestep{step}.model.log'
names = n
steps = [(i + 1) * 5000 for i in range(N)]

seeds = dict(
    softmax=[i+1 for i in range(3)],
    dtr_orig=[i + 1 for i in range(3)],
)
success = {k: list(list() for seed in seeds[k]) for k in seeds}

for k in seeds:
    for seed in seeds[k]:
        for step in steps:
            file_name = path.join(names[k].format(seed=seed, step=step))
            with open(file_name) as f:
                success[k][seed - 1].append(float(f.readlines()[-1].split()[-1]))

# %%
success_arr = {k: array(success[k]) for k in names}
# stats = ('min', 'max', 'median')
for k in names:
    plt.plot(
        array(steps) / 1e3, numpy.median(success_arr[k], 0),
        label=f'{k}',
        linewidth=2,
    )
for k in names:
    plt.fill_between(
        array(steps) / 1e3, success_arr[k].min(0), success_arr[k].max(0),
        alpha=.5,
    )
plt.grid(True)
plt.xlabel('steps [k–]')
plt.ylabel('success rate')
plt.legend(ncol=7)
plt.ylim([0.0, 0.85])
plt.xlim([5, 105])
plt.title('LogSumExp vs. Max')
plt.xticks(range(10, 100 + 10, 10));
plt.show()
#plt.savefig('Rgr-vs-hrd-success_rate-min-max.png', bbox_inches = 'tight')

# %%
N = 20
n = dict(); s = 'piecewise_lin'; d = 'dtr_orig'
n[s] = '/home/sk7685/ppuu_experiments/piecewise_lin/models/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed={seed}-novaluestep{step}.model.log'
n[d] = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed={seed}-novaluestep{step}.model.log'
names = n
steps = [(i + 1) * 5000 for i in range(N)]

seeds = dict(
    piecewise_lin=[i+1 for i in range(3)],
    dtr_orig=[i + 1 for i in range(3)],
)
success = {k: list(list() for seed in seeds[k]) for k in seeds}

for k in seeds:
    for seed in seeds[k]:
        for step in steps:
            file_name = path.join(names[k].format(seed=seed, step=step))
            with open(file_name) as f:
                success[k][seed - 1].append(float(f.readlines()[-1].split()[-1]))

# %%
success_arr = {k: array(success[k]) for k in names}
# stats = ('min', 'max', 'median')
for k in names:
    plt.plot(
        array(steps) / 1e3, numpy.median(success_arr[k], 0),
        label=f'{k}',
        linewidth=2,
    )
for k in names:
    plt.fill_between(
        array(steps) / 1e3, success_arr[k].min(0), success_arr[k].max(0),
        alpha=.5,
    )
plt.grid(True)
plt.xlabel('steps [k–]')
plt.ylabel('success rate')
plt.legend(ncol=7)
plt.ylim([0.50, 0.85])
plt.xlim([5, 105])
plt.title('Piecewise Linear versus Linear')
plt.xticks(range(10, 100 + 10, 10));
plt.show()
#plt.savefig('Rgr-vs-hrd-success_rate-min-max.png', bbox_inches = 'tight')

# %%
N = 20
n = dict(); s = 'nonlin_prox_mask'; d = 'dtr_orig'
n[s] = '/home/sk7685/ppuu_experiments/nonlin_prox_mask/models/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed={seed}-novaluestep{step}.model.log'
n[d] = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed={seed}-novaluestep{step}.model.log'
names = n
steps = [(i + 1) * 5000 for i in range(N)]

seeds = dict(
    nonlin_prox_mask=[i+1 for i in range(3)],
    dtr_orig=[i + 1 for i in range(3)],
)
success = {k: list(list() for seed in seeds[k]) for k in seeds}

for k in seeds:
    for seed in seeds[k]:
        for step in steps:
            file_name = path.join(names[k].format(seed=seed, step=step))
            with open(file_name) as f:
                success[k][seed - 1].append(float(f.readlines()[-1].split()[-1]))

# %%
success_arr = {k: array(success[k]) for k in names}
# stats = ('min', 'max', 'median')
for k in names:
    plt.plot(
        array(steps) / 1e3, numpy.median(success_arr[k], 0),
        label=f'{k}',
        linewidth=2,
    )
for k in names:
    plt.fill_between(
        array(steps) / 1e3, success_arr[k].min(0), success_arr[k].max(0),
        alpha=.5,
    )
plt.grid(True)
plt.xlabel('steps [k–]')
plt.ylabel('success rate')
plt.legend(ncol=7)
plt.ylim([0.50, 0.85])
plt.xlim([5, 105])
plt.title('Non-Linear versus Linear')
plt.xticks(range(10, 100 + 10, 10));
plt.show()
#plt.savefig('Rgr-vs-hrd-success_rate-min-max.png', bbox_inches = 'tight')

# %%
N = 20
n = dict(); s = 'const_slope'; d = 'dtr_orig'
n[s] = '/home/sk7685/ppuu_experiments/const_slope/models/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed={seed}-novaluestep{step}.model.log'
n[d] = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/planning_results/' + \
       'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2' + \
       '-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed={seed}-novaluestep{step}.model.log'
names = n
steps = [(i + 1) * 5000 for i in range(N)]

seeds = dict(
    const_slope=[i+1 for i in range(3)],
    dtr_orig=[i + 1 for i in range(3)],
)
success = {k: list(list() for seed in seeds[k]) for k in seeds}

for k in seeds:
    for seed in seeds[k]:
        for step in steps:
            file_name = path.join(names[k].format(seed=seed, step=step))
            with open(file_name) as f:
                success[k][seed - 1].append(float(f.readlines()[-1].split()[-1]))

# %%
success_arr = {k: array(success[k]) for k in names}
# stats = ('min', 'max', 'median')
for k in names:
    plt.plot(
        array(steps) / 1e3, numpy.median(success_arr[k], 0),
        label=f'{k}',
        linewidth=2,
    )
for k in names:
    plt.fill_between(
        array(steps) / 1e3, success_arr[k].min(0), success_arr[k].max(0),
        alpha=.5,
    )
plt.grid(True)
plt.xlabel('steps [k–]')
plt.ylabel('success rate')
plt.legend(ncol=7)
plt.ylim([0.10, 0.85])
plt.xlim([5, 105])
plt.title('Constant Slope vs. Original Deterministic FM')
plt.xticks(range(10, 100 + 10, 10));
plt.show()
#plt.savefig('Rgr-vs-hrd-success_rate-min-max.png', bbox_inches = 'tight')
