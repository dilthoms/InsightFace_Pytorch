from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('./')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = True
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'mobilefacenet' # or 'ir'
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'age'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100 # irse net depth 50 
    conf.batch_size = 256 # mobilefacenet
    conf.age_train_file = conf.data_path/'ageid.txt'
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
        conf.lam = 0.01
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-1
        conf.milestones = [4,6,7]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.restore = ''
        conf.ce_loss = CrossEntropyLoss()    
        conf.age_loss = MSELoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
