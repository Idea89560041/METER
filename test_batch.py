import torch
import torchvision
import models
import numpy as np
from PIL import Image



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def getresult(im_path):
    model_hyper = models.MeterNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    model_dti = models.DTINet(6, 224, 448, 224).cuda()
    model_dti.train(False)
    # load our pre-trained model
    model_hyper.load_state_dict((torch.load('./model/AQP_XXX.pth')))
    model_dti.load_state_dict((torch.load('./model/DTI_XXX.pth')))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

    img = pil_loader(im_path)
    img = transforms(img)
    img = torch.as_tensor(img.cuda()).unsqueeze(0)
    paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

    # Building target network
    model_target = models.TargetNet(paras).cuda()
    model_DTI = model_dti.cuda()
    for param in model_target.parameters():
        param.requires_grad = False
    for param in model_DTI.parameters():
        param.requires_grad = False

    # Quality prediction
    pred_type = model_DTI(paras['target_in_vec'])
    pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
    pred_q = torch.mul(pred_type, pred)
    pred_q = torch.sum(pred_q, dim=1, keepdim=False)
    score = pred_q.item()
    _, type = torch.max(pred_type.data, 1)
    print('Predicted quality score: %.2f' % score)
    print('Predicted type: %.2f' % type)

    return score, type

def to_excel(orgin_list, d_name):
    result = open('./test_data/{}_result.xls'.format(d_name), 'w', encoding='gbk')
    result.write('img_name\tscore\ttype\n')
    for m in range(len(orgin_list)):
        for n in range(len(orgin_list[m])):
            result.write(str(orgin_list[m][n]))
            result.write('\t')
        result.write('\n')
    result.close()

def generate_iqa_result(path, d_name):
    iqa_result = []
    for jpgfile in glob.glob(path):
        print(jpgfile)
        image_name = os.path.basename(jpgfile)
        score, type = getresult(jpgfile)
        iqa_result.append([image_name, score, type])
        to_excel(iqa_result, d_name)

if __name__ == '__main__':
    generate_iqa_result(r'./test_data/*', 'name')




