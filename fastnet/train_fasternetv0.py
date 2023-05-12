import os
import sys
import json



from comet_ml import Experiment
from torch import Tensor
from torchsampler import ImbalancedDatasetSampler



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

# optimizer
import math
# heatmap
import numpy as np
from PIL import Image,ImageOps
from utils import GradCAM, show_cam_on_image, center_crop_img
# from PIL import Image
# import matplotlib.pyplot as plt
# model
from model_v3 import mobilenet_v3_large
from comet_ml.integration.pytorch import log_model

# class weights
from collections import Counter 

# lion
from optimizer_lion import Lion
experiment = Experiment()

# fastnet add
from models.fasternet import fasternet_s
# mixup
from timm.data.mixup import Mixup

# TemporaryGrad
class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        torch.set_grad_enabled(self.prev)

# heatmap
def Gcam(model, tar_cate, image, img, device):
    
    # img = center_crop_img(img, 224)

    # [C, H, W]
    with TemporaryGrad():
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        input_tensor = torch.unsqueeze(img, dim=0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, target_category=int(tar_cate))

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.asarray(image).astype(dtype=np.float32) / 255.,
                                                    grayscale_cam,
                                                    use_rgb=True)
    return visualization


def learning_speed(val_acc: float, mode: str) -> float:
    if mode == "no_speed":
        return 1
    elif mode == "constant":
        return min(1 / (0.02 + val_acc), 15)
    elif mode == "log":
        return min(1 / (0.01 + math.log(val_acc + 1, 2)), 15)
    elif mode == "e":
        return 1 / (0.02 + math.e ** (val_acc - 1))
    elif mode == "root":
        return min(1 / (0.005 + val_acc ** 0.5), 15)
    
def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    

    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 16,
    "num_classes": 27,
    "alpha": 1,
    "input_size": 224,
    "mobilenet_name": fasternet_s,
    "loss_function_name": nn.CrossEntropyLoss,
    "optimizer_name": optim.AdamW,
    "ImbalancedDatasetSampler": False, # ImbalancedDatasetSampler or class_weights
    "class_weights": True,
    "learning_speed": "no_speed",
    "early_stop_patient": 10,
    "accumulation_step": 2,
    # "use_amp": True,
    "schedule": optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6),
    "pretrained": True,
    "use_ema": True,
    "model_ema_deca": 0.9998
    }
    patient_times = 0
    assert not (hyper_params["ImbalancedDatasetSampler"] and hyper_params["class_weights"]),"both_True"
    model_name_set="fasternet_s"
    experiment.log_parameters(hyper_params)
    exper_mame = "lr_{}_{}_{}".format(hyper_params["learning_rate"],
                         "optim.Adam",
                         hyper_params["learning_speed"]
                         )
    experiment.set_name(exper_mame)

    experiment.add_tags([
                model_name_set, 
                "CrossEntropyLoss",
                "optim.Adam",
                "class wight",
                "t_learn", 
                hyper_params["learning_rate"]
                ])

    # model
    model_name_set="mobilenet_v3_large"
    save_name = '/{}_{}_{}_{}.pth'.format(hyper_params["batch_size"], hyper_params["epochs"], 
                            hyper_params["alpha"], model_name_set)

    # path_load
    load_path = "/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/Test6_mobilenet/path.json"
    with open(load_path, "r", encoding="utf-8") as f:
        path_load = json.load(f)


    classjson_path = path_load["classjson_path"]
    root_path = path_load["root_path"]
    data_set_dir = path_load["data_set_dir"]
    flower_dir = path_load["flower_dir"]
    weight_path = path_load["weight_path"]
    train_save_path = path_load["train_save_path"] + save_name
    
    


    # ToTensor H，W，C ——> C，H，W C/255 [0~1]
    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomRotation(10),
                                    transforms.GaussianBlur(kernel_size=(5,5),sigma=(0.1, 3.0)),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    transforms.Resize(hyper_params["input_size"]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std= [0.18507297, 0.18050247, 0.16784933])
                                    ]),
        "val_test": transforms.Compose([
                                    transforms.Resize(hyper_params["input_size"]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std= [0.18507297, 0.18050247, 0.16784933])
                                    ])}
    mixup_fn = Mixup(
                    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
                    prob=0.1, switch_prob=0.5, mode='batch',
                    label_smoothing=0.1, num_classes=hyper_params["num_classes"])

    data_root = os.path.abspath(os.path.join(os.getcwd(), root_path))  # get data root path
    image_path = os.path.join(data_root, data_set_dir, flower_dir)  # flower data set path
    
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    

    # class weights
    if hyper_params["class_weights"]:
        label_nums_dic = Counter([j for i, j in train_dataset.imgs])
        sample_num_list = label_nums_dic.values()
        max_sample_num = max(label_nums_dic.values())
        weights = [max_sample_num/label_nums for label_nums in sample_num_list]
        class_weights = torch.FloatTensor(weights).to(device)




    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())


    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(classjson_path, 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), hyper_params["batch_size"] if hyper_params["batch_size"] > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # ImbalancedDatasetSampler
    if hyper_params["ImbalancedDatasetSampler"]:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=ImbalancedDatasetSampler(train_dataset),
                                                batch_size=hyper_params["batch_size"],
                                                num_workers=nw)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=hyper_params["batch_size"],
                                                shuffle=True,
                                                num_workers=nw)
    # val data
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val_test"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=hyper_params["batch_size"], shuffle=False,
                                                  num_workers=nw)
    # test data
    testdate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["val_test"])
    test_num = len(testdate_dataset)
    testdate_loader = torch.utils.data.DataLoader(testdate_dataset,
                                                batch_size=hyper_params["batch_size"], shuffle=False,                                  
                                                num_workers=nw)
        
    # UnNormalize tensor ToPILImage
    reback_img = transforms.Compose([
                transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])]
                )



    print("using {} images for training, {} images for validation., {} images for test".format(train_num, val_num, test_num))
    
    # create model
    net = hyper_params["mobilenet_name"](pretrained=hyper_params["pretrained"])
    num_fr=net.head.in_features
    net.head=nn.Linear(num_fr, hyper_params["num_classes"])
    best_acc = 0.0

    # net = mobilenet_v3_large(num_classes=hyper_params["num_classes"], alpha=hyper_params["alpha"])
    # Initialize and train your model
    # model = TheModelClass()
    # train(model)

    # Seamlessly log your Pytorch model
    
    log_model(experiment, net, model_name=model_name_set)

    if weight_path != "":
        # load pretrain weights
        # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
        model_weight_path = weight_path
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        
        # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

        # freeze features weights
        # for param in net.features.parameters():
        #     param.requires_grad = False

    net.to(device)
    
    
    # define loss function
    # class weight
    if hyper_params["class_weights"]:
        loss_function = hyper_params["loss_function_name"](weight=class_weights)
    else:
        loss_function = hyper_params["loss_function_name"]()
    
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = hyper_params["optimizer_name"](params, lr=hyper_params["learning_rate"])
    save_path = train_save_path
    train_steps = len(train_loader)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # eta_min lr最小可變化 T_max 週期
    if hyper_params["schedule"] != None:
        lr_scheduler = hyper_params["schedule"](optimizer=optimizer, T_max=20, eta_min=1e-6)
    # ema
    if hyper_params["use_ema"]:
        model_ema = ModelEma(
                            net,
                            decay=model_ema_decay,
                            device=device,
                            )
    else:
        model_ema=None

    # Print model's state_dict
    modelstr=""
    for param_tensor in net.state_dict():
        modelstr += str(param_tensor) + ", \t" + str(net.state_dict()[param_tensor].size()) + "\n"
    experiment.set_model_graph(modelstr,overwrite=False)
    
        

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])
    # Log model weights	
    weights = []	
    for name in net.named_parameters():	
        if 'weight' in name[0]:	
            weights.extend(name[1].detach().cpu().numpy().tolist())	
    experiment.log_histogram_3d(weights, step=0)



    for epoch in range(hyper_params["epochs"]):
        # train
        net.train()
        running_loss = 0.0
        correct = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # faster( 
            samples, targets = mixup_fn(images, labels)
            # )
            optimizer.zero_grad()
            # logits = net(images.to(device))
            logits = net(samples.to(device))
            # train result
            train_r = torch.max(logits, dim=1)[1]
            batch_correct = torch.eq(train_r,labels.to(device)).sum().item()
            batch_total = labels.size(0)
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # ema
            # if model_ema is not None:
            #     model_ema.update(net)

            

            # log traing_batch_loss
            experiment.log_metric("train_batch_loss", loss.item())
            
            # log train_batch_accuracy
            experiment.log_metric("train_batch_accuracy", batch_correct / batch_total)
            
            # statistics
            running_loss += loss.item()
            correct += batch_correct

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     hyper_params["epochs"],
                                                                     loss)
        # Log model weights	
        weights = []	
        for name in net.named_parameters():	
            if 'weight' in name[0]:	
                weights.extend(name[1].detach().cpu().numpy().tolist())	
        experiment.log_histogram_3d(weights, step=epoch + 1)

        experiment.log_metric("train_accuracy", correct / train_num )   

        
        
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        
        
        # image_list = list()
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           hyper_params["epochs"])
                # image_list = unnormalize(val_images,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                # confusion_matrix.compute_matrix(val_lablist, pred_list, images=image_list, image_shape=(224,224,3))
                # ConfusionMatrixCallbackReuseImages(experiment, val_lablist, pred_list, confusion_matrix)
            
        val_accurate = acc / val_num


        # learning_speed change by val_accurate
        optimizer.param_groups[0]["lr"] = hyper_params["learning_rate"] * learning_speed(val_accurate, hyper_params["learning_speed"])

        if hyper_params["schedule"] != None:
            hyper_params["schedule"]()

        # experiment.log_image(image_list,name=val_labels,image_shape=(224,224,3))
        # experiment.log_confusion_matrix(val_lebels_list, predict_y_list,images=image_list ,title="Confusion Matrix, Epoch #%d" %(epoch + 1),
        #                                 file_name="Confusion-Matrix-%03d.json"  %(epoch + 1),image_shape=(224,224,3))
        # 上傳到comet
        experiment.log_metrics({"accuracy": val_accurate, "loss": (running_loss / train_steps)}, epoch=epoch+1)
        experiment.log_metrics({"learning_rate": optimizer.param_groups[0]["lr"]}, epoch=epoch+1)
        
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        
            
    # log_model(experiment, net, model_name="mobilenet_v3_large")

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    
        
        acc = 0.0
        # with torch.no_grad():
            # confusion_matrix
        confusion_matrix = experiment.create_confusion_matrix(labels=list(cla_dict.values()),max_categories=27,max_examples_per_cell=1000,image_shape=(234, 702, 3))
        
        with torch.no_grad():
            test_bar = tqdm(testdate_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
            
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
                                
                # image_list = unnormalize(test_images,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                
                # test_images = reback_img(test_images)
                # to pil_image
                
                target_list = list()
                for pre_index, image in enumerate(test_images):
                    image_list = list()
                    # tesnsot -> pilimage
                    reback = reback_img(image)
                    img_pil = transforms.ToPILImage()(reback)
                    
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
                    # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
                    # input_tensor = torch.unsqueeze(image, dim=0)
                    
                    visualization_pre = Gcam(net, predict_y[pre_index], img_pil, image, device)
                    visualization_true = Gcam(net, test_labels[pre_index], img_pil, image, device)
                    visualization_pre = Image.fromarray(visualization_pre.astype('uint8'), 'RGB')
                    visualization_true = Image.fromarray(visualization_true.astype('uint8'), 'RGB')
                    image_list.append(img_pil)
                    image_list.append(visualization_pre)
                    image_list.append(visualization_true)
                    target = Image.new('RGB',(224 * 3 + 30, 224 + 10))
                    for i,im in enumerate(image_list):
                        im = ImageOps.expand(im, 20, '#ffffff')
                        target.paste(im, (224 * int(i), 0))
                    
                    target_list.append(target)
                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, hyper_params["epochs"])
                confusion_matrix.compute_matrix(test_labels, predict_y, images=target_list, image_shape=(224, 224, 3))
                # ConfusionMatrixCallbackReuseImages(experiment, test_labels, predict_y, confusion_matrix)

        experiment.log_confusion_matrix(
        matrix=confusion_matrix,
        title="Confusion Matrix, Epoch %d" % (epoch + 1),
        file_name="confusion-matrix-%03d.json" % (epoch + 1),
        )
        test_accurate = acc / test_num
        experiment.log_metrics({"test_accuracy": test_accurate}, epoch=epoch + 1)
        
        
        if val_accurate < best_acc:
            patient_times += 1
            if patient_times == hyper_params["early_stop_patient"]:
                break
        else:
            patient_times = 0

    print('Finished Training')	
    experiment.log_model("weight", f'./{save_name}')
    experiment.end()



if __name__ == '__main__':
    main()