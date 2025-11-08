import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model import Gradient_Descent
from data_loader.dataset import Phantom2dDatasettrain1, Phantom2dDatasetval1, Phantom2dDatasettest
from trainer.train_1k_noprompt import Trainer
from trainer.test_1k_noprompt import Tester
from head_HM import *
from config import get_config


if __name__ == '__main__':
    args = get_config()
    log(args) #加载参数
    # model
    model = Gradient_Descent(args)
    # main
    if args.phase == 'tr':
        tr_dataset = Phantom2dDatasettrain1(args, phase='tr', datadir=args.tr_dir, length=int(args.imagenum_train), angle=[120])
        vl_dataset = Phantom2dDatasetval1(args, phase='vl', datadir=args.vl_dir, length=int(args.imagenum_val),angle=[120])
        train = Trainer(args, model, tr_dset=tr_dataset, vl_dset=vl_dataset)
        train.tr()

    elif args.phase == 'test':
        test_dir = args.test_img_dir
        test_dset = Phantom2dDatasettest(args, phase='test', datadir=test_dir, length=int(args.imagenum_test), angle=[120])
        test = Tester(args, model, test_dset=test_dset)
        test.test()
    print('[*] Finish!')







