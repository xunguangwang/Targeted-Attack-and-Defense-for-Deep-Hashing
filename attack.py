import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.prototypenet import *
from utils.loss import *
from utils.data_provider import *
from utils.hamming_matching import *


def target_adv_loss(noisy_output, target_hash):
    loss = -torch.mean(noisy_output * target_hash)
    # loss = noisy_output * target_hash
    # loss = (loss -2)*loss
    # loss = torch.mean(loss)
    return loss

def target_hash_adv(model, query, target_hash, epsilon, step=1, iteration=2000, randomize=False):
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

    return query + delta.detach()


def sample_image(image, name, sample_dir='sample/attack'):
    image = image.cpu().detach()[2]
    image = transforms.ToPILImage()(image.float())
    image.save(os.path.join(sample_dir, name + '.png'), quality=100)



classes_dic = {'FLICKR-25K': 38, 'NUS-WIDE':21, 'MS-COCO': 80, 'ImageNet': 100, 'CIFAR-10': 10}
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
backbone = 'AlexNet'
batch_size = 32
bit = 32
epsilon = 8 / 255.0
iteration = 100

lr = 1e-4
transfer = False

dset_database = HashingDataset(DATA_DIR, DATABASE_FILE, DATABASE_LABEL)
dset_train = HashingDataset(DATA_DIR, TRAIN_FILE, TRAIN_LABEL)
dset_test = HashingDataset(DATA_DIR, TEST_FILE, TEST_LABEL)
database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)
num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

database_labels = load_label(DATABASE_LABEL, DATA_DIR)
train_labels = load_label(TRAIN_LABEL, DATA_DIR)
test_labels = load_label(TEST_LABEL, DATA_DIR)
target_labels = database_labels.unique(dim=0)


model_path = 'checkpoint/{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
model = load_model(model_path)
database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, model_name, backbone, bit)

if transfer:
    t_model_name = 'DPH'
    t_bit = 32
    t_backbone = 'ResNet18'
    t_model_path = 'checkpoint/{}_{}_{}_{}.pth'.format(dataset, t_model_name, t_backbone, t_bit)
    t_model = load_model(t_model_path)
else:
    t_model_name = model_name
    t_bit = bit
    t_backbone = backbone
t_database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, t_model_name, t_backbone, t_bit)
target_label_path = 'log/target_label_attack_{}.txt'.format(dataset)
test_code_path = 'log/test_code_{}_attack_{}.txt'.format(dataset, t_bit)


if os.path.exists(database_code_path):
    database_hash = np.loadtxt(database_code_path, dtype=np.float)
else:
    database_hash = generate_hash_code(model, database_loader, num_database, bit)
    np.savetxt(database_code_path, database_hash, fmt="%d")
if os.path.exists(t_database_code_path):
    t_database_hash = np.loadtxt(t_database_code_path, dtype=np.float)
else:
    t_database_hash = generate_hash_code(t_model, database_loader, num_database, t_bit)
    np.savetxt(t_database_code_path, t_database_hash, fmt="%d")
print('database hash codes prepared!')


pnet_path = 'checkpoint/PrototypeNet_{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
if os.path.exists(pnet_path):
    pnet = load_model(pnet_path)
else:
    pnet = PrototypeNet(bit, num_classes).cuda()
    optimizer_l = torch.optim.Adam(pnet.parameters(), lr=lr, betas=(0.5, 0.999))
    epochs = 100
    steps = 300
    # batch_size = 64
    lr_steps = epochs * steps
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    criterion_l2 = torch.nn.MSELoss()
    circle_loss = CircleLoss(m=0, gamma=1)

    # hash codes of training set
    B = generate_hash_code(model, train_loader, num_train, bit)
    B = torch.from_numpy(B).cuda()

    for epoch in range(epochs):
        for i in range(steps):
            select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

            optimizer_l.zero_grad()

            target_hash_l = pnet(batch_target_label)
            sp, sn = similarity(target_hash_l, B, batch_target_label, train_labels.cuda(), bit)
            logloss = circle_loss(sp, sn) / (batch_size)
            regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
            loss = logloss + regterm

            loss.backward()
            optimizer_l.step()
            if i % 30 == 0:
                print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}'.format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
            scheduler.step()

    torch.save(pnet, pnet_path)
    pnet.eval()


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


qB = np.zeros([num_test, t_bit], dtype=np.float32)
query_prototype_codes = np.zeros((num_test, bit), dtype=np.float)
perceptibility = 0
for it, data in enumerate(test_loader):
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
    query_adv = target_hash_adv(model, queries, prototype_codes, epsilon, iteration=iteration)

    perceptibility += F.mse_loss(queries, query_adv).data * batch_size_

    if transfer:
        query_code = t_model(query_adv)
    else:
        query_code = model(query_adv)
    query_code = torch.sign(query_code)
    qB[index.numpy(), :] = query_code.cpu().data.numpy()

    sample_image(queries, '{}_benign'.format(it))
    sample_image(query_adv, '{}_adv'.format(it))



np.savetxt(test_code_path, qB, fmt="%d")
print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
p_map = CalcMap(query_prototype_codes, t_database_hash, targeted_labels, database_labels.numpy())
print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % p_map)
t_map = CalcMap(qB, t_database_hash, targeted_labels, database_labels.numpy())
print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % t_map)
map = CalcTopMap(qB, database_hash, test_labels.numpy(), database_labels.numpy(), 5000)
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
map = CalcMap(qB, database_hash, test_labels.numpy(), database_labels.numpy())
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
