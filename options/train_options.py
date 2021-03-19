# -*- coding: utf-8 -*-
from .base_options import BaseOptions
import os


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument("--dataroot", type=str, default='/home/yifanc3/dataset/building/512_50p_over')
        parser.add_argument('--frame_name',type=str, default='DEM_retile')
        parser.add_argument('--mask_name',type=str, default='labels_retile')
        parser.add_argument('--lmbda', type=float, default=1e-6, help='weight regularization factor')
        parser.add_argument('--weight_init', type=str, default='he_normal')
        parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--num_filters', type=int, default=112, help='number of filters for the first conv')
        parser.add_argument('--filter_length', type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--pretrained_weights", type=str, default='')
        
        
        parser.add_argument('--pretrained', type=int, default=0)
        
#         parser.add_argument('--weight',type=float, default=200.0)
                            
#         self.isTrain = True
        return parser    

    def parse(self):
        opt = super().parse()
        opt.frame_path = os.path.join(opt.dataroot, opt.frame_name)
        opt.mask_path = os.path.join(opt.dataroot, opt.mask_name)
        
        self.print_options(opt)
        
        self.opt = opt
        return self.opt
