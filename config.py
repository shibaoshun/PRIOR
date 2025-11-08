import argparse
import numpy as np
class get_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='CT img Recon')
        self.parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--data_type', default='IMA', help='dcm, IMA')
        self.parser.add_argument('--resume', default=False, help='resume training')
        self.parser.add_argument('--manualSeed', type=int, default=205,help='manual seed')
        self.parser.add_argument('--log_dir', default='./result/logs/', help='tensorboard logs')

        self.parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
        self.parser.add_argument('--eta', type=float, default=5e-4, help='eta')
        self.parser.add_argument('--tau', type=float, default=1e-5, help='tau')
        self.parser.add_argument('--layers', type=int, default=15, help='stage')
        self.parser.add_argument('--lamda', type=float, default=5e-4, help='lamda')
        self.parser.add_argument("--milestone", type=int, default=[], nargs='+',
                                 help="When to decay learning rate")

        # Training Parameters
        self.parser.add_argument('--epoch', type=int, default=70, help='#epoch ')
        self.parser.add_argument('--tr_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--vl_batch', type=int, default=1, help='val batch size')
        self.parser.add_argument('--ts_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--test_model', default='grad_best.pth', help='dcm, png')

        self.parser.add_argument('--img_size', default=[512,512], help='image size')
        self.parser.add_argument('--sino_size', nargs='*', default=[360,800], help='sino size')
        self.parser.add_argument('--poiss_level',default=5e6, help='Poisson noise level')
        self.parser.add_argument('--gauss_level',default=[0.05], help='Gaussian noise level')
        self.parser.parse_args(namespace=self)


        self.mode= 'sparse'   #sparse,limited
        self.phase = 'test'   #'tr test'
        self.imagenum_train = '4000'
        self.imagenum_val = '400'
        self.imagenum_test = '500'

        # Result saving locations
        self.info = self.mode
        self.img_dir = './result/' + self.info + '/img/'
        self.model_dir = './result/' + self.info + '/ckp/'

        if self.phase == 'tr':
            print('!!!!!!训练步骤!!!!!!')
            print('This is mode %s' % self.mode)

            self.tr_dir = './data/train'
            self.vl_dir = './data/val'

            if self.resume == True:
                self.resume_ckp_dir = self.model_dir +'epoch11.pth'
                self.resume_ckp_resume = '11'

        elif self.phase == 'test':
            self.test_ckp_dir = self.model_dir + self.test_model
            print('!!!!!!测试步骤!!!!!!')
            print('This is mode %s' %self.mode)
            print(self.test_ckp_dir)
            self.test_save_img =False
            self.test_img_dir = './data/test'




