import os
import argparse
import pickle

# 監控
from comet_ml import Experiment

# 為了混淆矩陣
import sys
from tqdm import tqdm


import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import mobile_vit_small as create_model
from utils import read_split_data, train_one_epoch, evaluate

# 監控
from comet_ml.integration.pytorch import log_model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights") is False:
        os.makedirs("/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    
    # Initialize and train your model
    # model = TheModelClass()
    # train(model)

    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="mobile_vit_small")

    

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)

    

    best_model = create_model(num_classes=args.num_classes)
    best_acc = 0.0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        

        # 上傳到comit
        experiment.log_metrics({"train_loss": train_loss, "train_acc": train_acc}, epoch=epoch+1)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        # 上傳到comit
        experiment.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, epoch=epoch+1)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 上傳到comit
        experiment.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, epoch=epoch+1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = create_model(num_classes=args.num_classes)
            torch.save(model.state_dict(), "/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights/best_model.pth")

        torch.save(model.state_dict(), "/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights/latest_model.pth")

    # 混淆矩陣
    
    best_model.eval()
    val_lebels_list = list()
    predict_y_list = list()

    data_loader = tqdm(val_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        val_lebels_list += labels
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        predict_y_list += pred_classes
                                                   
    # 畫混淆矩陣
    experiment.log_confusion_matrix(val_lebels_list, predict_y_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=22)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 創建監控
    experiment = Experiment(
            api_key = "oRjh6XGWKwj5us9cwCQA1arVk",
            project_name = "classfiy-project",
            workspace="teams-project"
            )


    # 超參數紀錄
    experiment.set_name("My Project mobilevit")
    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
    "num_classes": 22,
    "learning_rate": 0.0002,
    "epochs": 30,
    "batch_size": 16,
    "total_last_epoch": 60
    }

    experiment.log_parameters(hyper_params)



    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/content/drive/MyDrive/model/deep-learning-for-image-processing-master/data_set/project_data/project_photes")
    
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights/latest_model_60.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
