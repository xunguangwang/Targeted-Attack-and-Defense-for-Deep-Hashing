import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.prototypenet import *
from model.backbone import *
from utils.loss import *
from utils.data_provider import *
from utils.hamming_matching import *


def target_adv_loss(noisy_output, target_hash):
    loss = -torch.mean(noisy_output * target_hash)
    # loss = noisy_output * target_hash
    # loss = (loss -2)*loss
    # loss = torch.mean(loss)
    return loss

def target_hash_adv(model, query, target_hash, epsilon, step=2, iteration=7, randomize=True):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        noisy_output = model(query + delta)
        loss = target_adv_loss(noisy_output, target_hash.detach())
        loss.backward()

        delta.data = delta - step/255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

        # if i % 1 == 0:
        #     print('it:{}, loss:{}'.format(i, loss))
    return query + delta.detach()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


classes_dic = {'FLICKR-25K': 38, 'NUS-WIDE':21, 'MS COCO': 80, 'ImageNet': 100, 'CIFAR-10': 10}
dataset = 'NUS-WIDE'
DATA_DIR = '../data/{}'.format(dataset)
DATABASE_FILE = 'database_img.txt'
TRAIN_FILE = 'train_img.txt'
TEST_FILE = 'test_img.txt'
DATABASE_LABEL = 'database_label.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'
num_classes = classes_dic[dataset]
model_name = 'DPH'
backbone = 'VGG11'
batch_size = 32
bit = 32
epsilon = 8 / 255.0
epochs = 100
iteration = 7

lr = 1e-4
weight_decay = 1e-5

dset_database = HashingDataset(DATA_DIR, DATABASE_FILE, DATABASE_LABEL)
dset_train = HashingDataset(DATA_DIR, TRAIN_FILE, TRAIN_LABEL)
dset_test = HashingDataset(DATA_DIR, TEST_FILE, TEST_LABEL)
database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)
num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

database_labels = load_label(DATABASE_LABEL, DATA_DIR)
train_labels = load_label(TRAIN_LABEL, DATA_DIR).cuda()
test_labels = load_label(TEST_LABEL, DATA_DIR)
target_labels = database_labels.unique(dim=0)


model_path = 'checkpoint/{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
robust_model_path = 'checkpoint/adv_{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
model = load_model(model_path)
database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, model_name, backbone, bit)
target_label_path = 'log/target_label_{}.txt'.format(dataset)


database_hash = generate_hash_code(model, database_loader, num_database, bit)
print('database hash codes prepared!')


# pnet_path = 'checkpoint/PrototypeNet_{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
# if os.path.exists(pnet_path):
#     pnet = load_model(pnet_path)
# else:
#     pnet = PrototypeNet(bit, num_classes).cuda()
#     optimizer_l = torch.optim.Adam(pnet.parameters(), lr=lr, betas=(0.5, 0.999))
#     epochs = 100
#     steps = 300
#     # batch_size = 64
#     lr_steps = epochs * steps
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
#     criterion_l2 = torch.nn.MSELoss()
#     circle_loss = CircleLoss(m=0, gamma=1)

#     # hash codes of training set
#     B = generate_hash_code(model, train_loader, num_train, bit)
#     B = torch.from_numpy(B).cuda()

#     for epoch in range(epochs):
#         for i in range(steps):
#             select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
#             batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

#             optimizer_l.zero_grad()

#             target_hash_l = pnet(batch_target_label)
#             sp, sn = similarity(target_hash_l, B, batch_target_label, train_labels.cuda(), bit)
#             logloss = circle_loss(sp, sn) / (batch_size)
#             regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
#             loss = logloss + regterm

#             loss.backward()
#             optimizer_l.step()
#             if i % 30 == 0:
#                 print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}'.format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
#             scheduler.step()

#     torch.save(pnet, pnet_path)
#     pnet.eval()


pnet = PrototypeNet(bit, num_classes).cuda()
pnet.train()
model.train()
circle_loss = CircleLoss(m=0, gamma=1)
optimizer_l = torch.optim.Adam(pnet.parameters(), lr=lr, betas=(0.5, 0.999))
# opt = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=weight_decay)
opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
lr_steps = epochs * len(train_loader)
scheduler_l = torch.optim.lr_scheduler.MultiStepLR(optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
B = generate_hash_code(model, train_loader, num_train, bit)
B = torch.from_numpy(B).cuda()
U_ben = torch.zeros(num_train, bit).cuda()
U_adv = torch.zeros(num_train, bit).cuda()
# adversarial training
for epoch in range(epochs):
    epoch_loss = 0.0
    for it, data in enumerate(train_loader):
        x, y, index = data
        x = x.cuda()
        y = y.cuda()
        batch_size_ = index.size(0)

        output_ben = model(x)
        B[index.numpy(), :] = torch.sign(output_ben.detach())

        select_index = np.random.choice(range(target_labels.size(0)), size=batch_size_)
        batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
        set_requires_grad(pnet, True)
        optimizer_l.zero_grad()
        target_hash_l = pnet(batch_target_label)
        sp, sn = similarity(target_hash_l, B, batch_target_label, train_labels.cuda(), bit)
        logloss = circle_loss(sp, sn) / (batch_size_)
        regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size_)
        loss_p = logloss + regterm
        loss_p.backward()
        optimizer_l.step()
        scheduler_l.step()

        batch_prototype_codes = target_hash_l
        prototype_codes = torch.sign(batch_prototype_codes)
        x_adv = target_hash_adv(model, x, prototype_codes, epsilon, step=2, iteration=iteration, randomize=True)

        set_requires_grad(pnet, False)
        model.zero_grad()
        output_adv = model(x_adv)
        for i, ind in enumerate(index):
            U_ben[ind, :] = output_ben.data[i]
            U_adv[ind, :] = output_adv.data[i]

        S = CalcSim(y, train_labels)
        S1 = (y.mm(train_labels.t()) > 0).float()
        # DPH
        theta_x = output_ben.mm(Variable(U_ben).t()) / bit
        logloss = (theta_x - S.cuda()) ** 2
        loss = logloss.sum() / (num_train * batch_size_)

        # DPSH
        # Bbatch = torch.sign(output_ben)
        # theta_x = output_ben.mm(Variable(U_ben.cuda()).t()) / 2
        # logloss = (Variable(S1.cuda()) * theta_x - log_trick(theta_x)).sum() / (num_train * batch_size_)
        # regterm = (Bbatch - output_ben).pow(2).sum() / (num_train * batch_size_)
        # loss = -logloss + 50 * regterm

        # HashNet
        # loss = pairwise_loss_updated(output_ben, U_ben.cuda(), y, train_labels)

        theta_x = output_adv.mm(Variable(U_ben).t()) / bit
        logloss = (theta_x - S.cuda()) ** 2
        loss += logloss.sum() / (num_train * batch_size_)
        loss.backward()
        opt.step()
        scheduler.step()

        if it % 30 == 0:
            print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, loss_p: {:.5f}, loss: {:.5f}'.format(epoch, it, scheduler.get_last_lr()[0], loss_p, loss))


# torch.save(pnet, pnet_path)
torch.save(model, robust_model_path)
pnet.eval()
model.eval()


if os.path.exists(target_label_path):
    targeted_labels = np.loadtxt(target_label_path, dtype=np.int)
else:
    targeted_labels = np.zeros([num_test, num_classes])
    for data in test_loader:
        _, label, index = data
        batch_size_ = index.size(0)

        select_index = np.random.choice(range(target_labels.size(0)), size=batch_size_)
        batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
        targeted_labels[index.numpy(), :] = batch_target_label.cpu().data.numpy()

    np.savetxt(target_label_path, targeted_labels, fmt="%d")


qB = np.zeros([num_test, bit], dtype=np.float32)
query_prototype_codes = np.zeros((num_test, bit), dtype=np.float)
perceptibility = 0
for data in test_loader:
    queries, _, index = data

    n = index[-1].item() + 1
    print(n)
    queries = queries.cuda()
    batch_size_ = index.size(0)

    batch_target_label = targeted_labels[index.numpy(), :]
    batch_target_label = torch.from_numpy(batch_target_label).float().cuda()

    batch_prototype_codes = pnet(batch_target_label)
    prototype_codes = torch.sign(batch_prototype_codes)
    query_prototype_codes[index.numpy(), :] = prototype_codes.cpu().data.numpy()
    query_adv = target_hash_adv(model, queries, prototype_codes, epsilon, iteration=10, randomize=False)

    perceptibility += F.mse_loss(queries, query_adv).data * batch_size_

    query_code = model(query_adv)
    qB[index.numpy(), :] = query_code.cpu().data.numpy()


print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
p_map = CalcMap(query_prototype_codes, database_hash, targeted_labels, database_labels.numpy())
print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % p_map)
t_map = CalcMap(qB, database_hash, targeted_labels, database_labels.numpy())
print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % t_map)
