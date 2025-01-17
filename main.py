import os
import copy
from matplotlib.pyplot import axis
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from logger import Logger
from datetime import datetime
from data import MovieLens1MColdStartDataLoader, TaobaoADColdStartDataLoader, YahooColdStartDataLoader
from model import FactorizationMachineModel, WideAndDeep, DeepFactorizationMachineModel, AdaptiveFactorizationNetwork, ProductNeuralNetworkModel
from model import AttentionalFactorizationMachineModel, DeepCrossNetworkModel, MWUF, MetaE, CVAR
from model.wd import WideAndDeep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
logger = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pretrain_model_path', default='')
    parser.add_argument('--pretrained_base_model_path', default='')
    # parser.add_argument('--dataset_name', default='taobaoAD', help='required to be one of [movielens1M, taobaoAD]')
    parser.add_argument('--dataset_name', default='movielens', help='required to be one of [movielens, yahoo]')
    # parser.add_argument('--datahub_path', default='./datahub/')
    parser.add_argument('--datahub_path', default='./datahub/WWW_data')
    parser.add_argument('--warmup_model', default='cvar', help="required to be one of [base, mwuf, metaE, cvar, cvar_init]")
    parser.add_argument('--is_dropoutnet', type=bool, default=False, help="whether to use dropout net for pretrain")
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help="dropout ratio")
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--model_name', default='deepfm', help='backbone name, we implemented [fm, wd, deepfm, afn, ipnn, opnn, afm, dcn]')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--cvar_epochs', type=int, default=2)
    parser.add_argument('--cvar_iters', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--runs', type=int, default=3, help = 'number of executions to compute the average metrics')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--content', default="all", help="required to be one of [all, video_only, text_only]")
    parser.add_argument('--log_file', default="")
    args = parser.parse_args()
    return args

def get_loaders(name, datahub_path, device, bsz, content_mode, shuffle):
    path = os.path.join(datahub_path, name, "{}_data.pkl".format(name))
    if name == 'movielens1M' or name == "movielens":
        dataloaders = MovieLens1MColdStartDataLoader(name, path, device, bsz=bsz, content_mode=content_mode, shuffle=shuffle)
    elif name == 'taobaoAD':
        dataloaders = TaobaoADColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    elif name == 'yahoo':
        dataloaders = YahooColdStartDataLoader(name, path, device, bsz=bsz, content_mode=content_mode, shuffle=shuffle)
    else:
        raise ValueError('unknown dataset name: {}'.format(name))
    return dataloaders

def get_model(name, dl):
    if name == 'fm':
        return FactorizationMachineModel(dl.description, 16)
    elif name == 'wd':
        return WideAndDeep(dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'deepfm':
        return DeepFactorizationMachineModel(dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(dl.description, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropout=0)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(dl.description, embed_dim=16, mlp_dims=(16, ), dropout=0, method='inner')
    elif name == 'opnn':
        return ProductNeuralNetworkModel(dl.description, embed_dim=16, mlp_dims=(16, ), dropout=0, method='outer')
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(dl.description, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'dcn':
        return DeepCrossNetworkModel(dl.description, embed_dim=16, num_layers=3, mlp_dims=[16, 16], dropout=0.2)
    return

def test(model, data_loader, device):
    model.eval()
    labels, scores, predicts = list(), list(), list()
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        for _, (features, label) in enumerate(tqdm(data_loader)):
            features = {key: value.to(device) for key, value in features.items()}
            label = label.to(device)
            y = model(features)
            labels.extend(label.tolist())
            scores.extend(y.tolist())
    scores_arr = np.array(scores)
    return roc_auc_score(labels, scores), f1_score(labels, (scores_arr > np.mean(scores_arr)).astype(np.float32).tolist())

def test_ranking(model, data_loader, device, topk=10):
    length = len(data_loader)
    sum_num_hit, precision, recall, ndcg = 0, 0, 0, 0
    tqdm_dataloader = tqdm(data_loader)
    for features, label in tqdm(tqdm_dataloader):
        features = {k: v.squeeze(0) for k, v in features.items()}
        features["count"] = torch.ones(len(features["count"]))
        features["count"] = features["count"].to(device)
        features["count"] = features["count"].unsqueeze(1)
        pos_items = torch.Tensor(label).cpu().numpy()
        num_pos = len(pos_items)

        score = model(features)
        item_indices = score.cpu().detach().numpy().argsort()[-topk:][::-1]
        top_items = features["item_id"].squeeze().cpu().detach().numpy()[item_indices]

        num_hit = len(np.intersect1d(pos_items, top_items))
        sum_num_hit += num_hit
        precision += float(num_hit / topk)
        recall += float(num_hit / min(num_pos, topk))

        ndcg_score = 0.0
        max_ndcg_score = 0.0
        for i in range(min(num_pos, topk)):
            max_ndcg_score += 1 / np.log2(i + 2)
        if max_ndcg_score == 0:
            continue
        for i, temp_item in enumerate(top_items):
            if temp_item in pos_items:
                ndcg_score += 1 / np.log2(i + 2)
        ndcg += ndcg_score / max_ndcg_score
    precision, recall, ndcg_score = precision / length, recall / length, ndcg / length
    return (precision, recall, ndcg_score)


def dropoutNet_train(model, data_loader, device, epoch, lr, weight_decay, save_dir, dropout_ratio, log_interval=10, val_data_loader=None):
    # train
    logger.info("TRAINING MODEL (DROPOUTNET) STARTS")
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)

    save_best_prec = os.path.join(save_dir, "best_prec_dropout.pth")
    save_best_recall = os.path.join(save_dir, "best_recall_dropout.pth")
    save_best_ndcg = os.path.join(save_dir, "best_ndcg_dropout.pth")
    best_prec, best_recall, best_ndcg = 0.0, 0.0, 0.0
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        tqdm_dataloader = tqdm(data_loader)
        for i, (features, label) in enumerate(tqdm_dataloader):
            bsz = label.shape[0]
            indices = np.arange(bsz)
            random.shuffle(indices)
            dropout_size = int(bsz * dropout_ratio)
            dropout_idx = indices[:dropout_size]
            item_emb = model.emb_layer["item_id"]
            origin_item_emb = item_emb(features["item_id"]).squeeze(1)
            mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True).repeat(dropout_size, 1).to(device)
            zero_item_emb = torch.zeros(dropout_size, origin_item_emb.shape[1]).to(device)
            if random.random() < 0.5:
                origin_item_emb[dropout_idx] = zero_item_emb
            else:
                origin_item_emb[dropout_idx] = mean_item_emb
            y = model.forward_with_item_id_emb(origin_item_emb, features)

            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            tqdm_dataloader.set_description(
                "Epoch {}, loss {:.3f} ".format(
                    epoch_i, loss.item()
                )
            )
        topk = 10
        precision, recall, ndcg = test_ranking(model, val_data_loader, device, topk)
        val_result = "Epoch {} prec@{}: {:.4f} rec@{}: {:4f} ndcg@{}: {:4f}".format(
            epoch_i, topk, precision, topk, recall, topk, ndcg
        )
        if precision > best_prec:
            logger.info("Save best precision model")
            torch.save(model, save_best_prec)
        if recall > best_recall:
            logger.info("Save best recall model")
            torch.save(model, save_best_recall)
        if ndcg > best_ndcg:
            logger.info("Save best ndcg model")
            torch.save(model, save_best_ndcg)
        logger.info(val_result)
    torch.save(model, os.path.join(save_dir, "latest_dropout.pth"))
    logger.info("TRAINING MODEL (DROPOUTNET) DONE")
    return 

def train(model, data_loader, device, epoch, lr, weight_decay, save_dir, log_interval=10, val_data_loader=None):
    # train
    logger.info("TRAINING MODEL STARTS")
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    save_best_auc = os.path.join(save_dir, "best_auc.pth")
    best_auc = 0.0
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        tqdm_dataloader = tqdm(data_loader)
        for i, (features, label) in enumerate(tqdm_dataloader):
            y = model(features)
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            tqdm_dataloader.set_description(
                "Epoch {}, loss {:.3f} ".format(
                    epoch_i, loss.item()
                )
            )
        auc, f1 = test(model, val_data_loader, device)
        if auc > best_auc:
            torch.save(model, save_best_auc)
        logger.info("Epoch {}/{} loss: {:.4f} val_auc: {:.4f} val_F1: {:.4f}".format(
            epoch_i, epoch, epoch_loss/total_iters, auc, f1))
    torch.save(model, os.path.join(save_dir, "latest_backbone.pth"))
    logger.info("TRAINING MODEL DONE")
    return 

def pretrain(dataset_name, 
         datahub_name,
         bsz,
         shuffle,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir,
         dropout_ratio,
         is_dropoutnet=False,
         content_mode="all"):
    device = torch.device(device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    logger.info("GET DATALOADER")
    dataloaders = get_loaders(dataset_name, datahub_name, device, bsz, content_mode, shuffle==1)
    logger.info("GET DATALOADER DONE")
    model = get_model(model_name, dataloaders).to(device)
    logger.info("="*20 + 'pretrain {}'.format(model_name) + "="*20)
    # init parameters
    model.init()
    # pretrain
    if is_dropoutnet:
        dropoutNet_train(model, dataloaders['warm_train'], device, epoch, lr, weight_decay, save_dir, dropout_ratio, val_data_loader=dataloaders['cold_val'])
    else:
        train(model, dataloaders['warm_train'], device, epoch, lr, weight_decay, save_dir, val_data_loader=dataloaders['warm_val'])
    logger.info("="*20 + 'pretrain {}'.format(model_name) + "="*20)
    return model, dataloaders

def base(model,
         dataloaders,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir):
    logger.info("*"*20 + "base" + "*"*20)
    device = torch.device(device)
    
    save_path = os.path.join(save_dir, 'model.pth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # data set list
    auc_list = []
    f1_list = []
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    for i, train_s in enumerate(dataset_list):
        auc, f1 = test(model, dataloaders['test'], device)
        auc_list.append(auc.item())
        f1_list.append(f1.item())
        logger.info("[base model] evaluate on [test dataset] auc: {:.4f}, F1 socre: {:.4f}".format(auc, f1))
        if i < 3:
            model.only_optimize_itemid()
            train(model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    logger.info("*"*20 + "base" + "*"*20)
    return auc_list, f1_list

def base_test(model,
        dataloaders,
        model_name,
        epoch,
        lr,
        weight_decay,
        device,
        save_dir):
    logger.info("*"*20 + "base" + "*"*20)
    device = torch.device(device)
    
    save_path = os.path.join(save_dir, 'model.pth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # data set list
    result = {}
    topk_list = [1, 5, 10, 20]
    for topk in topk_list:
        result[topk] = test_ranking(model, dataloaders["cold_test"], device, topk=topk)
    return result

def metaE(model,
          dataloaders,
          model_name,
          epoch,
          lr,
          weight_decay,
          device,
          save_dir):
    logger.info("*"*20 + "metaE" + "*"*20)
    device = torch.device(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    metaE_model = MetaE(model, warm_features=dataloaders.item_features, device=device).to(device)
    # fetch data
    metaE_dataloaders = [dataloaders[name] for name in ['metaE_a', 'metaE_b', 'metaE_c', 'metaE_d']]
    # train meta embedding generator
    metaE_model.train()
    criterion = torch.nn.BCELoss()
    metaE_model.optimize_metaE()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, metaE_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        dataloader_a = metaE_dataloaders[epoch_i]
        dataloader_b = metaE_dataloaders[(epoch_i + 1) % 4]
        epoch_loss = 0.0
        total_iter_num = len(dataloader_a)
        iter_dataloader_b = iter(dataloader_b)
        for i, (features_a, label_a) in enumerate(dataloader_a):
            features_b, label_b = next(iter_dataloader_b)
            loss_a, target_b = metaE_model(features_a, label_a, features_b, criterion)
            loss_b = criterion(target_b, label_b.float())
            loss = 0.1 * loss_a + 0.9 * loss_b
            metaE_model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if (i + 1) % 10 == 0:
                logger.info("    iters {}/{}, loss: {:.4f}, loss_a: {:.4f}, loss_b: {:.4f}".format(i + 1, int(total_iter_num), loss, loss_a, loss_b), end='\r')
        logger.info("Epoch {}/{} loss: {:.4f}".format(epoch_i, epoch, epoch_loss/total_iter_num), " " * 100)
    # replace item id embedding with warmed itemid embedding
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = metaE_model.model.emb_layer[metaE_model.item_id_name].weight.data
        warm_item_id_emb = metaE_model.warm_item_id(features)
        indexes = features[metaE_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    # test by steps 
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    f1_list = []
    for i, train_s in enumerate(dataset_list):
        logger.info("#"*10, dataset_list[i],'#'*10)
        train_s = dataset_list[i]
        auc, f1 = test(metaE_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        f1_list.append(f1.item())
        logger.info("[metaE] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            metaE_model.model.only_optimize_itemid()
            train(metaE_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    logger.info("*"*20 + "metaE" + "*"*20)
    return auc_list, f1_list

def mwuf(model,
         dataloaders,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir):
    logger.info("*"*20 + "mwuf" + "*"*20)
    device = torch.device(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    # train mwuf
    mwuf_model = MWUF(model, 
                      item_features=dataloaders.item_features,
                      train_loader=train_base,
                      device=device).to(device)
    
    mwuf_model.init_meta()
    mwuf_model.train()
    criterion = torch.nn.BCELoss()
    mwuf_model.optimize_new_item_emb()
    optimizer1 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mwuf_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    mwuf_model.optimize_meta()
    optimizer2 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mwuf_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    mwuf_model.optimize_all()
    total_iters = len(train_base)
    loss_1, loss_2 = 0.0, 0.0
    for i, (features, label) in enumerate(train_base):
        # if i + 1 > total_iters * 0.3:
        #     break
        y_cold = mwuf_model.cold_forward(features)
        cold_loss = criterion(y_cold, label.float())
        mwuf_model.zero_grad()
        cold_loss.backward()
        optimizer1.step()
        y_warm = mwuf_model.forward(features)
        warm_loss = criterion(y_warm, label.float())
        mwuf_model.zero_grad()
        warm_loss.backward()
        optimizer2.step()
        loss_1 += cold_loss
        loss_2 += warm_loss
        if (i + 1) % 10 == 0:
            logger.info("    iters {}/{}  warm loss: {:.4f}" \
                    .format(i + 1, int(total_iters), \
                     warm_loss.item()), end='\r')
    logger.info("final average warmup loss: cold-loss: {:.4f}, warm-loss: {:.4f}"
                    .format(loss_1/total_iters, loss_2/total_iters))
    # use trained meta scale and shift to initialize embedding of new items
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = mwuf_model.model.emb_layer[mwuf_model.item_id_name].weight.data
        warm_item_id_emb = mwuf_model.warm_item_id(features)
        indexes = features[mwuf_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    # test by steps 
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    f1_list = []
    for i, train_s in enumerate(dataset_list):
        logger.info("#"*10, dataset_list[i],'#'*10)
        train_s = dataset_list[i]
        auc, f1 = test(mwuf_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        f1_list.append(f1.item())
        logger.info("[mwuf] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            mwuf_model.model.only_optimize_itemid()
            train(mwuf_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    logger.info("*"*20 + "mwuf" + "*"*20)
    return auc_list, f1_list

def cvar(model,
       dataloaders,
       model_name,
       epoch,
       cvar_epochs,
       cvar_iters,
       lr,
       weight_decay,
       device,
       save_dir,
       only_init=False):
    logger.info("*"*20 + "cvar" + "*"*20)
    device = torch.device(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    # train cvar
    warm_model = CVAR(model, 
                    warm_features=dataloaders.item_features,
                    train_loader=train_base,
                    device=device).to(device)
    warm_model.init_cvar()
    def warm_up(dataloader, epochs, iters, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        warm_model.optimize_cvar()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, warm_model.parameters()), \
                                                lr=lr, weight_decay=weight_decay)
        batch_num = len(dataloader)
        # train warm-up model
        for e in range(epochs):
            for i, (features, label) in enumerate(dataloader):
                a, b, c, d = 0.0, 0.0, 0.0, 0.0
                for _ in range(iters):
                    target, recon_term, reg_term  = warm_model(features)
                    main_loss = criterion(target, label.float())
                    loss = main_loss + recon_term + 1e-4 * reg_term
                    warm_model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    a, b, c, d = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item()
                a, b, c, d = a/iters, b/iters, c/iters, d/iters
                if logger and (i + 1) % 10 == 0:
                    logger.info("    Iter {}/{}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}" \
                            .format(i + 1, batch_num, a, b, c, d), end='\r')
        # warm-up item id embedding
        train_a = dataloaders['train_warm_a']
        for (features, label) in train_a:
            origin_item_id_emb = warm_model.model.emb_layer[warm_model.item_id_name].weight.data
            warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
            indexes = features[warm_model.item_id_name].squeeze()
            origin_item_id_emb[indexes, ] = warm_item_id_emb
    warm_up(train_base, epochs=1, iters=cvar_iters, logger=True)
    # test by steps 
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list, f1_list = [], []
    for i, train_s in enumerate(dataset_list):
        logger.info("#"*10, dataset_list[i],'#'*10)
        train_s = dataset_list[i]
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        f1_list.append(f1.item())
        logger.info("[cvar] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            if not only_init:
                warm_up(dataloaders[train_s], epochs=cvar_epochs, iters=cvar_iters, logger=False)
    logger.info("*"*20 + "cvar" + "*"*20)
    return auc_list, f1_list

def cvar_simple(model,
        dataloaders,
        model_name,
        epoch,
        cvar_epochs,
        cvar_iters,
        lr,
        weight_decay,
        device,
        save_dir,
        only_init=False,
        logging=False):
    logger.info("*"*20 + "cvar" + "*"*20)
    device = torch.device(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # train cvar
    logger.info("TRAINING CVAR")
    train_base = dataloaders['warm_train']
    warm_model = CVAR(model, 
                    warm_features=dataloaders.item_features,
                    train_loader=train_base,
                    device=device).to(device)
    warm_model.init_cvar()
    warm_model.train()
    criterion = torch.nn.BCELoss()
    warm_model.optimize_cvar()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, warm_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    batch_num = len(train_base)
    epochs = cvar_epochs if not only_init else 1
    test_dataset_name = "cold_val"
    test_loader = dataloaders[test_dataset_name]
    save_best_prec = os.path.join(save_dir, "best_prec_cvar.pth")
    save_best_recall = os.path.join(save_dir, "best_recall_cvar.pth")
    save_best_ndcg = os.path.join(save_dir, "best_ndcg_cvar.pth")
    best_prec, best_recall, best_ndcg = 0.0, 0.0, 0.0
    for e in range(epochs):
        tqdm_dataloader = tqdm(train_base)
        for i, (features, label) in enumerate(tqdm_dataloader):
            a, b, c, d = 0.0, 0.0, 0.0, 0.0
            for _ in range(cvar_iters):
                target, recon_term, reg_term  = warm_model(features)
                main_loss = criterion(target, label.float())
                loss = main_loss + recon_term + 1e-4 * reg_term
                warm_model.zero_grad()
                loss.backward()
                optimizer.step()
                a, b, c, d = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item()
            a, b, c, d = a/cvar_iters, b/cvar_iters, c/cvar_iters, d/cvar_iters
            tqdm_dataloader.set_description(
                "Epoch {}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}".format(
                    e + 1, a, b, c, d
                )
            )

        result = cvar_test(test_dataset_name, test_loader, model, warm_model, device)
        precision, recall, ndcg = result[10]
        if precision > best_prec:
            logger.info("Save best precision model")
            torch.save(warm_model, save_best_prec)
        if recall > best_recall:
            logger.info("Save best recall model")
            torch.save(warm_model, save_best_recall)
        if ndcg > best_ndcg:
            logger.info("Save best ndcg model")
            torch.save(warm_model, save_best_ndcg)
        logger.info("[Epoch {}] loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}".format(e + 1, a, b, c, d))
        logger.info("[Epoch {}] evaluate on [{} dataset] p: {:.4f}, r: {:.4f} n: {:.4f}".format(e + 1, test_dataset_name, precision, recall, ndcg))

    # TEST WITH COLD_TEST
    save_latest = os.path.join(save_dir, 'latest_cvar.pth')
    torch.save(warm_model, save_latest)
    test_dataset_name = "cold_test"
    test_loader = dataloaders[test_dataset_name]
    return cvar_test(test_dataset_name, test_loader, model, warm_model, device)
    # val_result = "[base model] evaluate on [cold test dataset] prec@k: {:.4f} rec@k: {:4f} ndcg@k: {:4f}".format(
    #     precision, recall, ndcg
    # )
    # logger.info(val_result)
    # return (precision, recall, ndcg)

def cvar_test(dataset_name, test_loader, model, warm_model, device, topk=10):
    model_v = copy.deepcopy(model).to(device)
    logger.info(f"EVAL WITH {dataset_name}")
    # warm-up item id embedding (inference)
    logger.info("WARM-UP ITEM ID EMBEDDING")
    for (features, label) in test_loader:
        features = {k: v.squeeze(0) for k, v in features.items()}
        features["count"] = torch.ones(len(features["count"]))
        features["count"] = features["count"].to(device)
        features["count"] = features["count"].unsqueeze(1)
        origin_item_id_emb = model_v.emb_layer[warm_model.item_id_name].weight.data
        warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
        indexes = features[warm_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    
    logger.info("VALID WITH WARMED-UP EMBEDDINGS")
    topk_list = [1, 5, 10, 20]
    result = {}
    for topk in topk_list:
        result[topk] = test_ranking(model_v, test_loader, device, topk)
    return result

def run(model, dataloaders, args, model_name, warm):
    if warm == 'base':
        # auc_list, f1_list = base(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
        result = base_test(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'mwuf':
        auc_list, f1_list = mwuf(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'metaE': 
        auc_list, f1_list = metaE(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'cvar': 
        # auc_list, f1_list = cvar(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr, args.weight_decay, args.device, args.save_dir)
        result = cvar_simple(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'cvar_init': 
        # auc_list, f1_list = cvar(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr, args.weight_decay, args.device, args.save_dir, only_init=True)
        result = cvar_simple(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr, args.weight_decay, args.device, args.save_dir, only_init=True)
    return result

if __name__ == '__main__':
    args = get_args()
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    res = {}

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(BASE_DIR, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logfile_name = args.log_file if args.log_file else f"{args.model_name}_{args.warmup_model}_{start_time}.log"
    log_file = os.path.join(log_dir, logfile_name)
    Logger.initialize(log_file)
    logger = Logger.get_logger()
    logger.info("*"*20 + "ENVIRONMENT" + "*"*20)
    for arg, value in args._get_kwargs():
        logger.info(f"{arg}: {value}")
    logger.info("*"*50)    

    if args.content not in ["all", "video_only", "text_only"]:
        logger.error(f"Invalid option for content-mode: {args.content}")
        raise Exception
    
    torch.cuda.empty_cache()
    if args.mode == "train":
        # load or train pretrain models
        model_path = args.pretrain_model_path
        pretrain_dir = os.path.dirname(args.pretrain_model_path)
        if args.pretrain_model_path and not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir, exist_ok=True)
        print(model_path)
        if os.path.exists(model_path):
            logger.info(f"LOAD PRETRAINED BACKBONE MODEL: {args.model_name}")
            model = torch.load(model_path).to(args.device)
            dataloaders = get_loaders(args.dataset_name, args.datahub_path, args.device, args.bsz, args.content, args.shuffle==1)
        else:
            logger.info(f"TRAIN BACKBONE MODEL: {args.model_name}")
            model, dataloaders = pretrain(args.dataset_name, args.datahub_path, args.bsz, args.shuffle, args.model_name, \
                args.epoch, args.lr, args.weight_decay, args.device, args.save_dir, args.dropout_ratio, args.is_dropoutnet, args.content)
            if len(args.pretrain_model_path) > 0:
                torch.save(model, model_path)
                
        # warmup train and test
        logger.info("WARMUP TRAIN STARTS")
        model_v = copy.deepcopy(model).to(args.device)
        result = run(model_v, dataloaders, args, args.model_name, args.warmup_model)
        for k, v in result.items():
            precision, recall, ndcg = v    
            logger.info("P@{} {:4f} R@{} {:4f} NDCG@{} {:4f}".format(k, precision, k, recall, k, ndcg))
        
    elif args.mode == "test":
        logger.info(f"TEST {args.pretrain_model_path}")
        dataloaders = get_loaders(args.dataset_name, args.datahub_path, args.device, args.bsz, args.content, shuffle=False)
        topk_list = [1, 5, 10, 20]

        if args.warmup_model == "cvar":
            warm_model = torch.load(args.pretrain_model_path).to(args.device)
            model = torch.load(args.pretrained_base_model_path).to(args.device)
            result = cvar_test("cold_test", dataloaders["cold_test"], model, warm_model, args.device)
            for k, v in result.items():
                precision, recall, ndcg = v    
                logger.info("P@{} {:4f} R@{} {:4f} NDCG@{} {:4f}".format(k, precision, k, recall, k, ndcg))
        else:
            model = torch.load(args.pretrain_model_path).to(args.device)
            for topk in topk_list:
                precision, recall, ndcg = test_ranking(model, dataloaders["cold_test"], args.device, topk=topk)
                logger.info("P@{} {:4f} R@{} {:4f} NDCG@{} {:4f}".format(topk, precision, topk, recall, topk, ndcg))