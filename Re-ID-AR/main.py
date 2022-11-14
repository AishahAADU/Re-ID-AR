from __future__ import print_function, absolute_import
import torch
from torch.autograd import Variable
import numpy as np
from utils import *
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torchvision.transforms as T
from MARS_dataset import *
from LPW_dataset import *
from scheduler import *
from loss import *
from utils import *
from samplers import *
from video_loader import *
from model import Re_ID_AR
import argparse

def make_optimizer(model):
    params = []
    base_learning_rate =0.00035  # 0.000015 #0.00035
    weight_decay = 0.0005
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_learning_rate
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
  
    #if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':optimizer = getattr(torch.optim, "Adam")(params, momentum=0.9)else:
    optimizer = getattr(torch.optim, "Adam")(params)
    return optimizer


def test(model, queryloader, galleryloader,num_action, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    top1 = AverageMeter()
    top5 = AverageMeter()
    preds_tensor = np.empty(shape=[0, int(num_action)], dtype=np.byte)   # shape = (num_sample, num_Action)
    labels_tensor = np.empty(shape=[0, int(num_action)], dtype=np.byte)   # shape = (num_sample, num_Action)
    with torch.no_grad():
      all_target = []
      accuracy_list = []
      
      for batch_idx, (imgs, pids, camids,action) in enumerate(queryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        assert(b==1)
        imgs = imgs.view(b*n, s, c, h, w)
        features,action_score = model(imgs)
        
        a=[]
        for i in action:
                a.append(i)
        labels=a
        
        labels=torch.stack(tuple(labels))
        
        labels = to_one_hot(labels, C=int(num_action)).cuda() #one hot label
        
        labels = labels.cpu().numpy()
        
        preds = torch.gt(action_score, torch.ones_like(action_score)/2)
        preds=preds.cpu().numpy()
        preds_tensor = np.append(preds_tensor, preds, axis=0)
        labels_tensor = np.append(labels_tensor, labels, axis=0)
        
        _, preds = torch.max(action_score, 1)
            
        if batch_idx == 0:
                      all_predicted = preds
                      all_targets = action
        else:
                      all_predicted = torch.cat((all_predicted, preds),0)
                      all_targets = torch.cat((all_targets,action),0)

          
        features = features.data.cpu()
        action=action.cuda()
        
        prec1, prec5 = accuracy(action_score, action, topk=(1, 2))
        top1.update(prec1.item(), action.size(0))
        top5.update(prec5.item(), action.size(0))   
        
        #---------------------------------
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)

      qf = torch.cat(qf,dim=0)
      qf=qf.view(qf.size(0),2048) 
      q_pids = np.asarray(q_pids)
      q_camids = np.asarray(q_camids)


      print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
      gf, g_pids, g_camids = [], [], []
      for batch_idx, (imgs, pids, camids,action) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s , c, h, w)
        assert(b==1)
        features,action_score = model(imgs)

        a=[]
        for i in action:
                a.append(i)
           
        labels=a
        labels=torch.stack(tuple(labels))
        labels = to_one_hot(labels, C=int(num_action)).cuda()
        
        labels = labels.cpu().numpy()
        preds = torch.gt(action_score, torch.ones_like(action_score)/2)
        preds = preds.cpu().numpy()
        
        preds_tensor = np.append(preds_tensor, preds, axis=0)
        labels_tensor = np.append(labels_tensor, labels, axis=0)
        _, preds = torch.max(action_score, 1)

         
        all_predicted = torch.cat((all_predicted, preds),0)
        all_targets = torch.cat((all_targets,action),0)
        
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
        
        action=action.cuda()
        prec1, prec2 = accuracy(action_score, action, topk=(1, 2))
        top1.update(prec1.item(), action.size(0))
        top5.update(prec5.item(), action.size(0))   
        
       
    gf =torch.cat(gf ,dim=0)
    gf=gf.view(gf.size(0), 2048)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------- ")
    
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve")


    numberofaction=3
    accuracy_list=[]
    precision_list = []
    recall_list = []
    f1_score_list = []
    average_precision = 0.0
    average_recall = 0.0
    average_f1score = 0.0
    valid_count = 0

    for i in range(numberofaction): #action_list
       y_true, y_pred = labels_tensor[:, i], preds_tensor[:, i]
       
       accuracy_list.append(accuracy_score(y_true, y_pred))
       precision_list.append(precision_score(y_true, y_pred, average='binary'))
       recall_list.append(recall_score(y_true, y_pred, average='binary'))
       f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
       average_precision += precision_list[-1]
       average_recall += recall_list[-1]
       average_f1score += f1_score_list[-1]
       valid_count += 1

    average_acc = np.mean(accuracy_list)
    average_precision = average_precision / s   #valid_count  the only recognise ones are 5
    average_recall = average_recall / valid_count
    average_f1score = average_f1score / valid_count
    
    
    table = PrettyTable(['attribute', 'accuracy', 'precision', 'recall', 'f1 score'])
    name=0
    for i in range(numberofaction):
        table.add_row([name,
               '%.3f' % accuracy_list[i],
               '%.3f' % precision_list[i] if precision_list[i] >= 0.0 else '-',
               '%.3f' % recall_list[i] if recall_list[i] >= 0.0 else '-',
               '%.3f' % f1_score_list[i] if f1_score_list[i] >= 0.0 else '-',
               ])
        name+=1       
    print(table)
    print("average_acc new method::",average_acc)
    print("average average_precision method::",average_precision)
    print("average_recall new method::",average_recall)
    print("average_f1score new method::",average_f1score)
    print("Action accuracy",top1.avg)  
    
    target_names=[0,1,2]
    conf_mat_dict={}

    for label_col in range(len(target_names)):
        y_true_label = labels_tensor[:, label_col]
        y_pred_label = preds_tensor[:, label_col]
        conf_mat_dict[target_names[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

   
    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)

    target_names=[0,1,2]
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())


    
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=target_names, normalize=True, title= "plot")
    #plt.savefig(os.path.join("/home2/zwjx97/MarsAction",str(e)+"Actions.png"))
    print("multi-label confusion metrex ")
    print(multilabel_confusion_matrix(labels_tensor, preds_tensor))
    plt.close()
    return cmc[0], mAP

def to_one_hot(x, C=2, tensor_class=torch.FloatTensor):
    """ One-hot a batched tensor of shape (B, ...) into (B, C, ...) """
    x_one_hot = tensor_class(x.size(0), C, *x.shape[1:]).zero_()
    x_one_hot = x_one_hot.scatter_(1, x.unsqueeze(1), 1)
    return x_one_hot


__factory = {
    'MARS':MARS,
    'LPW':LPW,
}

if __name__ == '__main__':
    device = torch.device("cuda" )
    transform_train = T.Compose([
            T.Resize((224, 112), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([224, 112]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
        ])
           
    transform_test = T.Compose([
       T.Resize((224, 112), interpolation=3),
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    parser = argparse.ArgumentParser(description="Re-ID-AR")
    parser.add_argument(
        "--Dataset_name", default="", help="The name of the DataSet", type=str)
    parser.add_argument(
        "--Net_path", default="", help="pretrained model", type=str)

    parser.add_argument(
        "--action_num", default="", help="number of action to train the model with", type=str) 

    parser.add_argument(
        "--train_file", default="", help="train_file", type=str) 

    parser.add_argument(
        "--test_file", default="", help="test_file", type=str) 

    parser.add_argument(
        "--save_path", default="", help="model save path", type=str)     

    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    pretrainpath=args.Net_path
    save_path=args.save_path

    action=args.action_num
    train_file =args.train_file
    test_file =args.test_file

    dataset = __factory[Dataset_name](train_file,test_file )
    pin_memory = True

    trainloader = DataLoader(
    VideoDataset_inderase(dataset.train,dataset.train_action, seq_len=4, sample='intelligent',transform=transform_train,errize=True),
    sampler=RandomIdentitySampler(dataset.train),
    batch_size=64, num_workers=2,
    pin_memory=pin_memory, drop_last=True,
)
    
    queryloader = DataLoader(
    VideoDataset(dataset.query,dataset.query_action, seq_len=4, sample='dense',transform=transform_test),
    batch_size=1, num_workers=1,
    pin_memory=pin_memory, drop_last=False,
)

    galleryloader = DataLoader(
    VideoDataset(dataset.gallery,dataset.gallery_action, seq_len=4, sample='dense',transform=transform_test),
    batch_size=1, num_workers=1,
    pin_memory=pin_memory, drop_last=False,
)
    
   
    print('End dataloader...\n')
    
    # 1. Criterion
    
    criterion_RLL=RankedLoss(1.3,2.0,1.)
    center_criterion = CenterLoss(use_gpu=True,feat_dim=2048)   # for person identification
    criterion_cent_f = CenterLoss(num_classes=3, feat_dim=512, use_gpu=True) # for action recognition adjust number of action to 5, 3 or 8 
    cetner_loss_weight = 0.0005
    criterion_bce = nn.BCELoss().cuda()
    
    no_of_classes = 3
    beta = 0.9999
    gamma = 2.0
    #samples_per_cls = [1/794,1/40,1/406]
    if Dataset_name=='MARS':
        if int(action)==3:
                 samples_per_cls = [1/1592.0,1/80.0,1/828.0] # Three actions
        elif int(action)==5:         
            samples_per_cls = [1/396.0,1/20.0,1/14.0,1/33.0,1/159.0] # Five actions
        else:    
            samples_per_cls = [1/396.0,1/20.0,1/5.0,1/33.0,1/151.0,1/14.0,1/3.0,1/3.0]   # Eight actions

    elif Dataset_name=='LPW':
        if int(action)==3:
                 samples_per_cls = [1/1145.0,1/1934.0,1/23] # Three actions
        
        else:    
            samples_per_cls = [1/1145.0,1/948.0,1/372.0,1/23.0,1/268.0,1/23,1/88.0,1/14.0]   # Eight actions

    

    new_wighted=CEL_Sigmoid(np.asarray(samples_per_cls))


    # The model
    model = Re_ID_AR(model_name = 'resnet50_ibn_a',num_classes=625,last_stride=1,model_path=pretrainpath,  pretrain_choice= 'imagenet',Action_class=int(action)).to(device)
    


    # 2. Optimizer                                                 
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5) # for RE_id center 
    optimizer_center_a = torch.optim.SGD(criterion_cent_f.parameters(), lr=0.1) # for action center
    optimizer = make_optimizer(model)
   
    # 3. scheduler
    scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=10)
    
    
   



    id_loss_list = []
    trip_loss_list = []
    track_id_loss_list = []
    Action_list=[]
    lr_step_size=50
    best_cmc = 0
    
    for e in range(200):
        print('Epoch',e)
        
        scheduler.step()
        # test the model
        if ((e+1)%10== 0) :
             cmc,map = test(model, queryloader, galleryloader,int(action))
             print('CMC: %.4f, mAP : %.4f'%(cmc,map))
             if cmc >= best_cmc:
                torch.save(model.state_dict(),os.path.join(save_path,'MARS_ckpt_best.pth'))
                best_cmc = cmc
               
                
        total_id_loss = 0 
        total_RLL_loss = 0 
        total_track_id_loss = 0
        total_Action=0
        pbar = tqdm(total=len(trainloader),ncols=100,leave=True)
        model.train()
        
        for batch_idx, (imgs, pids, _, labels2,actions_lable) in enumerate(trainloader):
            
            criterion_ID = CrossEntropyLabelSmooth(len(pids)).cuda()
            seqs, labels = imgs.cuda(), pids.cuda()
            center_label=actions_lable.cuda()
            labels2=labels2.cuda()
            a=[]
            for i in actions_lable:
                a.append(i)
            
            actions_lable=a
            actions_lable=torch.stack(tuple(actions_lable))
            
            actions_lable = to_one_hot(actions_lable, C=int(action)).cuda()
            
            
            classf,feat,a_vals,action_score,f_a,fa_id= model(seqs) 
           
            # encourage the role of random erasing 
            attn_noise  = a_vals * labels2
            attn_loss = attn_noise.sum(1).mean()
            
            #Re-ID losses
            RLL=criterion_RLL(feat,labels).cuda()
    
            id_loss = criterion_ID(classf,labels) 
            center= center_criterion(feat,labels )
            RE_id_loss=  id_loss+ .0005 * center+RLL+attn_loss

            #Action loss
            seq_len=torch.Tensor([(32,1)])
            action_center=CENTERLOSS(f_a, action_score, actions_lable, seq_len, criterion_cent_f , e, device)
            action_ce=criterion_bce(action_score,actions_lable) # OR new_wighted(action_score,actions_lable)
            action_loss=action_ce+.0005*action_center


            
            total_id_loss += id_loss.item()
            total_RLL_loss+=RLL.item()
            
           
            
            # comulative loss
            total_loss =   RE_id_loss+  action_loss
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            optimizer_center_a.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            for param in center_criterion.parameters():
                param.grad.data *= (1./cetner_loss_weight)
               
            optimizer_center.step()
            pbar.update(1)
        pbar.close()
        

        
        print("total_loss",total_loss)
        avg_Action_loss = '%.4f'%(total_Action/len(trainloader))
        print("action avreg loss:",avg_Action_loss)
        avg_id_loss = '%.4f'%(total_id_loss/len(trainloader))
        avg_RLL_loss = '%.4f'%(total_RLL_loss/len(trainloader))
        avg_track_id_loss = '%.4f'%(total_track_id_loss/len(trainloader))
        print('RLL : %s , ID : %s , Track_ID : %s'%(avg_RLL_loss,avg_id_loss,avg_track_id_loss))
        
        
        
        
