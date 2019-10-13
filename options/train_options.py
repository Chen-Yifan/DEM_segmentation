from .base_options import BaseOptions
import os


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument("--dataroot", type=str, default='/home/yifanc3/dataset/data/selected_128_overlap')
        parser.add_argument('--frame_name',type=str, default='all_frames_5m6b_norm')
        parser.add_argument('--mask_name',type=str, default='all_masks_5m6b')
        parser.add_argument('--loss_function',type=str, default='single')
        parser.add_argument('--weight',type=float, default=200.0)
        parser.add_argument('--load_pretrained', type=int, default=0)
        
        
                            
        self.isTrain = True
        return parser    

    def parse(self):
        opt = super().parse()
        opt.frame_path = os.path.join(opt.dataroot, opt.frame_name)
        opt.mask_path = os.path.join(opt.dataroot, opt.mask_name)
        
        self.print_options(opt)
        
        self.opt = opt
        return self.opt
