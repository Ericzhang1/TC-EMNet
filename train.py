import argparse
import logging
import os
import sys
import timeit
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tqdm import tqdm, trange

from model import TC_EMNet
from dataset import *

import torch
import torch.nn as nn

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler)
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score

from evaluation import *
#set up logger for outputing logs and writer for tensorboardX
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
tb_writer = None

def get_args():
    parser = argparse.ArgumentParser(description='Deep learning model for disease progression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument("--save_dir", type=str, default="",
                        help="directory for saving the model")
    parser.add_argument("--data_dir", type=str, default="",
                        help="directory for raw data")
    parser.add_argument("--output_dir", type=str, default="",
                        help="directory for output results")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to do evaluation")
    parser.add_argument("--model_dir", type=str, default="",
                        help="Model directory for loading")
    parser.add_argument('--seed', type=int, default=25,
                        help='random seed')
    parser.add_argument('--eval_epochs', type=int, default=10,
                        help='interval in epochs for doing evaluation during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--silent', action='store_true',
                        help="whether to show progress bar for tqdm")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help="regularization term")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="dropout probability for model")
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='hidden size for the network')                    
    parser.add_argument('--adapting_epoch', type=int, default=5,
                        help='epochs to adaptively adjust the epoch.')
    parser.add_argument('--fold', type=int, default=0,
                     help='fold to run with training')  
    parser.add_argument('--test_epochs', type=int, default=10,
                    help='epochs to adaptively adjust the epoch.')
    parser.add_argument('--data_type', type=int, default=0,
                    help='data types for corresponding test, 0 for all, 1 for new cell, 2 for new drug, 3 for both')
    parser.add_argument('--cross_validate', action='store_true',
                        help="Whether to cross validate models")
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers of the model')
    parser.add_argument('--clustering', action='store_true',
                        help="wheter to do clustering")
    parser.add_argument('--hop', type=int, default=0,
                        help="number of hops for memory network")
    parser.add_argument('--alpha', type=float, default=0.3,
                        help="number of hops for memory network")
    parser.add_argument('--use_label', action='store_true',
                        help="wheter to use label during training")
    parser.add_argument('--label_embed', type=int, default=32,
                        help="target embedding dimension")
    
    return parser.parse_args()

def train(model, 
          device,
          train_dataset, 
          validate_dataset,
          test_dataset,
          args):
    try:
        os.mkdir(args.save_dir)
        logging.info('Created model directory')
    except OSError:
        pass

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batchsize)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
    if args.clustering:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_sampler))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Trainable parameters = %d", pytorch_total_params)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.batchsize * args.gradient_accumulation_steps)

    model.zero_grad()
    train_iterator = trange(int(args.epochs), desc="Epoch", disable=args.silent)
    epoch, global_step = 0, 0
    best_val, curr_val = -1, 0
    best_test, curr_test = -1, 0
    decay_strike = 0

    def kl_anneal_function(step, k=0.25, x0=700): #500
        return min(1, step/x0)

    def kl_loss(mean, std, global_step):
        KL_loss = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())
        KL_weight = kl_anneal_function(global_step)
        return KL_loss, KL_weight

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.silent, position=0, leave=True)
        epoch += 1
        epoch_loss = 0
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            visit, label, length = batch['patient'], batch['label'], batch['length']
            visit, label, length = visit.to(device=device), label.to(device=device), length.to(device=device)
            
            output, hidden_, mean, std, recon = model(visit, label)
            loss = 0
            for i, l in enumerate(batch['length']):
                if args.clustering:
                    pred, gt = output[i, 0:l, :], visit[i, 0:l, :]
                    loss += criterion(pred, gt)
                else:
                    pred, gt = output[i, 0:l, :], label[i, 0:l, :]
                    gt = torch.tensor(np.where(gt.cpu()==1)[1])
                    gt = gt.to(device=device)
                    loss += criterion(pred, gt)
          
                for i, l in enumerate(batch['length']):
                    mean_, std_ = mean[i, 0:l, :], std[i, 0:l, :]
                    k_loss, kl_weight = kl_loss(mean_, std_, global_step)
                    loss += k_loss * kl_weight

                    pred, gt = recon[i, 0:l, :], visit[i, 0:l, :]
                    loss += args.alpha * criterion2(pred, gt)
            loss /= args.batchsize
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            tb_writer.add_scalar('gradient_norm', total_norm, global_step)

            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 60)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                #add loss to tensorboardX
                tb_writer.add_scalar('loss', loss*args.gradient_accumulation_steps, global_step)
            
        #evalution during training
        if epoch % args.eval_epochs == 0:
            #evaluate(model, device, validate_dataset, args)
            try:
                if args.clustering:
                    eval_loss, purity, nmi, ri = evaluate(model, device, validate_dataset, args)
                else:
                    eval_loss, auc, purity, nmi, ri = evaluate(model, device, validate_dataset, args)
            except:
                #model has yet ready for prediction
                logger.info('Evaluation skipped')
                continue
            
            tb_writer.add_scalar('eval_loss', eval_loss, global_step)
            if args.clustering:
                tb_writer.add_scalar('auc', purity, global_step)
                tb_writer.add_scalar('nmi', nmi, global_step)
                tb_writer.add_scalar('ri', ri, global_step)
            else:
                tb_writer.add_scalar('auc', purity, global_step)
                tb_writer.add_scalar('nmi', nmi, global_step)
                tb_writer.add_scalar('ri', ri, global_step)
                tb_writer.add_scalar('auc', auc, global_step)
            logger.info('-' * 100)
            if args.clustering:
                log_str = '| Eval {:3d} at epoch {:>8d} | val: {:7.4f} '\
                            '| purity: {:7.4f} | NMI: {:7.4f}  | RI: {:7.4f}'.format(epoch // args.eval_epochs, epoch, eval_loss, purity, nmi, ri)
            else:
                log_str = '| Eval {:3d} at epoch {:>8d} | val: {:7.4f} '\
                            '| auc: {:7.4f} | purity: {:7.4f}  | NMI: {:7.4f}  | RI: {:7.4f}'.format(epoch // args.eval_epochs, epoch, eval_loss, auc, purity, nmi, ri)
            logger.info(log_str)
            logger.info('-' * 100)
            if args.clustering:
                curr_val = purity
            else:
                curr_val = auc
            #save the model with best auc
            if curr_val >= best_val:
                torch.save(model.state_dict(),
                        args.save_dir + f'/train_fold{args.fold}_data{args.data_type}_{args.seed}.pth')
                logger.info(f'model saved !')
                best_val = curr_val
                decay_strike = 0
            else:
                decay_strike += 1
            if decay_strike == args.adapting_epoch:
                for param_group in optimizer.param_groups:
                    print('Learning rate was: ', param_group['lr'])
                    param_group['lr'] *= 0.1
                    print('Learning rate decayed to: ', param_group['lr'])
                decay_strike = 0

        if epoch % args.test_epochs == 0:
            try:
                if args.clustering:
                    eval_loss, purity, nmi, ri = evaluate(model, device, test_dataset, args)
                else:
                    eval_loss, auc, purity, nmi, ri = evaluate(model, device, test_dataset, args)
            except:
                #model has yet ready for prediction
                logger.info('Testing skipped')
                continue
            tb_writer.add_scalar('test_loss', eval_loss, global_step)
            if args.clustering:
                tb_writer.add_scalar('purity_test', purity, global_step)
                tb_writer.add_scalar('NMI_test', nmi, global_step)
                tb_writer.add_scalar('RI_test', ri, global_step)
            else:
                tb_writer.add_scalar('purity_test', purity, global_step)
                tb_writer.add_scalar('NMI_test', nmi, global_step)
                tb_writer.add_scalar('RI_test', ri, global_step)
                tb_writer.add_scalar('test_p', auc, global_step)
            logger.info('-' * 100)
            if args.clustering:
                log_str = '| Test {:3d} at epoch {:>8d} | val: {:7.4f} '\
                            '| purity: {:7.4f} | NMI: {:7.4f}  | RI: {:7.4f}'.format(epoch // args.test_epochs, epoch, eval_loss, purity, nmi, ri)
            else:
                log_str = '| Eval {:3d} at epoch {:>8d} | val: {:7.4f} '\
                            '| auc: {:7.4f} | purity: {:7.4f}  | NMI: {:7.4f}  | RI: {:7.4f}'.format(epoch // args.eval_epochs, epoch, eval_loss, auc, purity, nmi, ri)
            logger.info(log_str)
            logger.info('-' * 100)
            if args.clustering:
                curr_test = purity
            else:
                curr_test = auc
            #save the model with best auc
            if curr_test >= best_test:
                torch.save(model.state_dict(),
                        args.save_dir + f'/test_fold_{args.fold}_data{args.data_type}_{args.seed}.pth')
                logger.info(f'test model saved !')
                best_test = curr_test
                
        logger.info("  Epoch = {}, loss = {}".format(epoch, epoch_loss/(step+1)))
       
    tb_writer.close()

    #run test
    model.load_state_dict(
        torch.load(args.save_dir + f'/test_fold{args.fold}_data{args.data_type}_{args.seed}.pth', map_location=device)
    )
    logger.info(f'Model loaded from {args.save_dir}')
    args.do_train = False
    evaluate(model, device, test_dataset, args)
    if not args.cross_validate:
        model.load_state_dict(
            torch.load(args.save_dir + f'/test_fold{args.fold}_data{args.data_type}_{args.seed}.pth', map_location=device)
        )
        logger.info(f'Model loaded from {args.save_dir}')
        evaluate(model, device, test_dataset, args)

def evaluate(model, 
          device,
          dataset,
          args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batchsize)

    start_time = timeit.default_timer()
    if not args.do_train:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.batchsize)
    
    eval_loss, step = 0, 0
    if args.clustering:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    activation = nn.Softmax(dim=1)
    pred = torch.tensor([], dtype=torch.float32, device=device)
    ground_truth = torch.tensor([], dtype=torch.float32, device=device)
    hidden = torch.tensor([], dtype=torch.float32, device=device)
    ground_truth_cluster = torch.tensor([], dtype=torch.float32, device=device)
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.silent, position=0, leave=True):
        model.eval()
        with torch.no_grad():
            visit, label, length = batch['patient'], batch['label'], batch['length']
            visit, label, length = visit.to(device=device), label.to(device=device), length.to(device=device)
            output, hidden_, mean, std, recon = model(visit, label)
            batch_loss = 0
            for i, l in enumerate(batch['length']):
                if args.clustering:
                    pred_, gt, hid = output[i, 0:l, :], visit[i, 0:l, :], hidden_[i, 0:l, :]
                    diag = label[i, 0:l, :]
                    label_ = torch.tensor(np.where(diag.cpu()==1)[1])
                    label_ = label_.to(device=device)
                    loss = criterion(pred_, gt)
                    pred = torch.cat([pred, pred_.float()], dim=0)
                    ground_truth = torch.cat([ground_truth, label_.float()], dim=0)
                    hidden = torch.cat([hidden, hid.float()], dim=0)
                else:
                    pred_, gt, hid = output[i, 0:l, :], label[i, 0:l, :], hidden_[i, 0:l, :]
                    label_ = torch.tensor(np.where(gt.cpu()==1)[1])
                    label_ = label_.to(device=device)
                    loss = criterion(pred_, label_)
                    pred = torch.cat([pred, activation(pred_.float())], dim=0)
                    ground_truth = torch.cat([ground_truth, gt.float()], dim=0)
                    hidden = torch.cat([hidden, hid.float()], dim=0)
                    ground_truth_cluster = torch.cat([ground_truth_cluster, label_.float()], dim=0)
                batch_loss += loss.item()
            batch_loss /= args.batchsize
            eval_loss += batch_loss 
            step += 1

    pred = pred.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    hidden = hidden.cpu().numpy()
    ground_truth_cluster = ground_truth_cluster.cpu().numpy()
    eval_loss /= step
    #np.save(f'{args.output_dir}/gt.npy', ground_truth)
    #np.save(f'{args.output_dir}/pred.npy', pred)
    #calculate auc for multi-label prediction 
    if args.clustering:
        purity, nmi, ri = k_means(args, hidden, ground_truth)
    else:   
        purity, nmi, ri = k_means(args, hidden, ground_truth_cluster)
        auc = roc_auc_score(ground_truth, pred, average='samples')
    if not args.do_train:
        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
        logger.info("***** Evaluation Results *****")
        if args.clustering:
            logger.info(f'purity: {purity} (higher is better, max=1)')
            logger.info(f'NMI: {nmi} (higher is better, max=1)')
            logger.info(f'ri: {ri} (higher is better, max=1)')
        else:
            logger.info(f'AUC: {auc} (higher is better, max=1)')
            logger.info(f'purity: {purity} (higher is better, max=1)')
            logger.info(f'NMI: {nmi} (higher is better, max=1)')
            logger.info(f'ri: {ri} (higher is better, max=1)')
    
        #np.save(f'{args.output_dir}/gt.npy', ground_truth)
        #np.save(f'{args.output_dir}/pred.npy', pred)
    
    np.save(f'{args.output_dir}/hidden_{args.fold}_{args.data_type}_{args.seed}.npy', hidden)
    if args.clustering:
        return eval_loss, purity, nmi, ri
    else:
        return eval_loss, auc, purity, nmi, ri
    
if __name__ == '__main__':
    args = get_args()

    #set up log
    if not args.cross_validate:
        log_dir = f'{args.output_dir}/log'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tb_writer = SummaryWriter(log_dir)
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    logger.info(f'loading dataset ')

    dataset = disease_progression(data_dir=args.data_dir, args=args)
    
    train_dataset = disease_progression_aux(dataset, train=True, valid=False, test=False, seed=args.seed, fold=args.fold)
    valid_dataset = disease_progression_aux(dataset, train=False, valid=True, test=False, seed=args.seed, fold=args.fold)
    test_dataset = disease_progression_aux(dataset, train=False, valid=False, test=True, seed=args.seed, fold=args.fold)

    logger.info(f'initializing model: TC_EMNet')
    args.device = device
    model = TC_EMNet(args)

    if args.model_dir:
        model.load_state_dict(
            torch.load(args.model_dir, map_location=device)
        )
        logger.info(f'Model loaded from {args.model_dir}')
    model.to(device=device)

    if args.do_train:
        #evaluate(model, device, test_dataset, args)
        train(model, device, train_dataset, valid_dataset, test_dataset, args)
    if args.do_eval:
        evaluate(model, device, test_dataset, args)
    
    if args.cross_validate:
        result, result2, result3 = [], [], []
        for fold in range(5):
            args.fold = fold
            args.do_train = True
            #set up log
            log_dir = f'{args.output_dir}/log_fold{args.fold}_{args.data_type}_{args.seed}'
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            tb_writer = SummaryWriter(log_dir)
            
            train_dataset = disease_progression_aux(dataset, train=True, valid=False, test=False, seed=args.seed, fold=args.fold)
            valid_dataset = disease_progression_aux(dataset, train=False, valid=True, test=False, seed=args.seed, fold=args.fold)
            test_dataset = disease_progression_aux(dataset, train=False, valid=False, test=True, seed=args.seed, fold=args.fold)
            train(model, device, train_dataset, valid_dataset, test_dataset, args)
            model.load_state_dict(
                torch.load(args.save_dir + f'/test_fold{args.fold}_data{args.data_type}_{args.seed}.pth', map_location=device)
            )
            logger.info(f'Model loaded from {args.save_dir}')
            args.do_train = False
            if args.clustering:
                eval_loss, purity, nmi, ri = evaluate(model, device, test_dataset, args)
            else:
                eval_loss, auc, purity, nmi, ri = evaluate(model, device, test_dataset, args)
            #auc += auc_
            result.append(purity)
            result2.append(nmi)
            result3.append(ri)
            model = TC_EMNet(args)
            model.to(device=device)
            
        logger.info('=' * 100)
        for i in range(5):
            p = result[i]
            nmi = result2[i]
            ri = result3[i]
            logger.info(f'Fold {i}: {p}')
            logger.info(f'Fold {i}: {nmi}')
            logger.info(f'Fold {i}: {ri}')
            logger.info('=' * 50)

        f_p, f_p_std = np.mean(result), np.std(result)
        f_n, f_n_std = np.mean(result2), np.std(result2)
        f_r, f_r_std = np.mean(result3), np.std(result3)

        save = np.asarray([np.mean(result), np.mean(result2), np.mean(result3)])
        np.save(f'{args.output_dir}/result_{args.seed}_{args.data_type}.npy', save)
        logger.info(f'Final purity is {f_p} +/- {f_p_std}')
        logger.info(f'Final nmi is {f_n} +/- {f_n_std}')
        logger.info(f'Final ri is {f_r} +/- {f_r_std}')
        logger.info('=' * 100)
    