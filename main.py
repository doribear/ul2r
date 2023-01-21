from UL2R import UL2R_dataset, MLM_dataset, Classify_dataset
from transformers import AlbertForMaskedLM, AlbertConfig, AlbertForSequenceClassification, get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn import metrics

import os
import torch
import deepspeed
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', default = 'mlm', type = str, help = 'running type [mlm, flan, cls]')
parser.add_argument('-p', '--plm', default = '', type = str, help = 'pre-trained model path when [mode == cls]')
parser.add_argument('-s', '--save_path', default = 'result/train_result', type = str, help = 'train result save path')
parser.add_argument('--local_rank')

parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()


torch.manual_seed(100)


def get_mlm_settings():
    config = AlbertConfig(30000, hidden_size = 512, num_attention_heads = 8)
    model = AlbertForMaskedLM(config)
    dataset = MLM_dataset.fromfile()
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 10000)

    return model, dataset, lossfn, optimizer, scheduler


def get_UL2R_settings():
    config = AlbertConfig(30000, hidden_size = 512, num_attention_heads = 8)
    model = AlbertForMaskedLM(config)
    dataset = UL2R_dataset.fromfile()
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 10000)

    return model, dataset, lossfn, optimizer, scheduler

def get_classification_settings():
    dataset = Classify_dataset.fromfile()
    config = AlbertConfig(30000, hidden_size = 512, num_attention_heads = 8, num_labels = len(dataset.label_dict))
    model = AlbertForSequenceClassification(config)
    model.load_state_dict(torch.load(args.plm, 'cpu')['module'], strict = False)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch : (1 - 1e-4) ** epoch)
    return model, dataset, lossfn, optimizer, scheduler

def get_paths(save_path):
    paths = save_path.split('/')

    for i in range(1, len(paths)):
        paths[i] = os.path.join(paths[i-1], paths[i]) 

    for path in paths:
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)

def pretrain(model, dataset, lossfn, optimizer, scheduler):

    get_paths(args.save_path)
    model, optimizer, loader, scheduler = deepspeed.initialize(model = model, 
    optimizer = optimizer, 
    model_parameters = model.parameters(), 
    training_data = dataset, 
    lr_scheduler = scheduler, 
    config = {'train_batch_size' : 32, 'train_micro_batch_size_per_gpu' : 16})

    writer = SummaryWriter(f'{args.save_path}/sub_node_result')
    writer.add_text('parameter_size', f'{sum(p.numel() for p in model.parameters())}', 0)
    for step, (ids, mask, type_ids, attention_mask) in tqdm(enumerate(loader), desc = 'training', total = len(loader)):
        optimizer.zero_grad()
        ids, mask, type_ids, attention_mask = ids.to(model.device), mask.to(model.device), type_ids.to(model.device), attention_mask.to(model.device)
        out = model(input_ids = mask, token_type_ids = type_ids, attention_mask = attention_mask)['logits']
        loss = lossfn(out.view(out.shape[0] * out.shape[1], -1), ids.flatten())
        model.backward(loss)
        optimizer.step()
        scheduler.step()
        writer.add_scalar('loss', loss.item(), step)
        if step % 1000 == 0:
            model.save_checkpoint(args.save_path, args.mode)
    model.save_checkpoint(args.save_path, args.mode)


def fine_tuning_classification_task(model, dataset, lossfn, optimizer, scheduler):

    get_paths(args.save_path)
    model, optimizer, loader, scheduler = deepspeed.initialize(model = model, 
    optimizer = optimizer, 
    model_parameters = model.parameters(), 
    training_data = dataset, 
    lr_scheduler = scheduler, 
    config = {'train_batch_size' : 32, 'train_micro_batch_size_per_gpu' : 16})

    writer = SummaryWriter(f'{args.save_path}/sub_node_result')
    writer.add_text('parameter_size', f'{sum(p.numel() for p in model.parameters())}', 0)
    for epoch in range(200):
        result_metrics =  {'accuracy' : [], 'recall' : [], 'precision' : [], 'f1_macro' : []}
        losses = []
        for ids, type_ids, attention_mask, label in tqdm(loader, desc = 'training'):
            optimizer.zero_grad()
            ids, type_ids, attention_mask, label = ids.to(model.device), type_ids.to(model.device), attention_mask.to(model.device), label.to(model.device)
            out = model(input_ids = ids, token_type_ids = type_ids, attention_mask = attention_mask)['logits']
            loss = lossfn(out.view(out.shape[0], -1), label.flatten())
            model.backward(loss)
            optimizer.step()
            out, label = out.argmax(-1).flatten().detach().cpu().numpy(), label.flatten().detach().cpu().numpy()
            losses.append(loss.item())
            result_metrics['accuracy'].append(metrics.accuracy_score(label, out))
            result_metrics['recall'].append(metrics.recall_score(label, out, average = 'micro'))
            result_metrics['precision'].append(metrics.precision_score(label, out, average = 'micro'))
            result_metrics['f1_macro'].append(metrics.f1_score(label, out, average = 'macro'))
        for key in result_metrics.keys():
            result_metrics[key] = np.mean(result_metrics[key])
        writer.add_scalars('metric', result_metrics, epoch)
        writer.add_scalar('loss', np.mean(losses), epoch)
        writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
        model.save_checkpoint(args.save_path, args.mode)
        scheduler.step()


if args.mode == 'mlm':
    model, dataset, lossfn, optimizer, scheduler = get_mlm_settings()
    pretrain(model, dataset, lossfn, optimizer, scheduler)
elif args.mode == 'flan':
    model, dataset, lossfn, optimizer, scheduler = get_UL2R_settings()
    pretrain(model, dataset, lossfn, optimizer, scheduler)
elif args.mode == 'cls':
    model, dataset, lossfn, optimizer, scheduler = get_classification_settings()
    fine_tuning_classification_task(model, dataset, lossfn, optimizer, scheduler)
