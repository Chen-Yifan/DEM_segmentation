import argparse
import os
from util import util

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        
        parser.add_argument("--date", type=str, default='tryout', help='date is a split directory to save our model')
        parser.add_argument("--isTrain", action='store_true', help='is True, in train mode, otherwise test mode')
        parser.add_argument("--model", type=str, default='unet', help='choose which model to use')
        parser.add_argument("--results_dir", type=str, default='./results/', help='results are saved here')
        parser.add_argument('--ckpt_name', type=str, default='experiment', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='models are saved here')
        parser.add_argument('--augmentation', action='store_true', help='do augmentation or not')
        parser.add_argument("--n_epoch", type=int, default=100)
        parser.add_argument("--optimizer", type=str, default='Adam', help='1:Adam, 2:Adadelta')
        parser.add_argument('--input_channel', type=int, default=1)
        parser.add_argument('--dim', type=int, default=128, help='input image width/height')
        parser.add_argument('--use_gradient', action='store_true', help='if use gradient as input')
        parser.add_argument('--DEM_only', action='store_true', help='if only use DEM as input')
        parser.add_argument('--n_classes', type=int, default=2, help='output classes')
        parser.add_argument('--save_model', action='store_true', help='whether or not we save the model to ckpt_dir')
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--loss', type=str, default='bce', help='which loss function to use')
        parser.add_argument('--visualize',action='store_true', help='if save visualization results')
        parser.add_argument('--dataset', type=str, default='erosion', help='erosion/building')
        

        self.initialized = True
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are difined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.ckpt_dir, opt.date, opt.ckpt_name)
        util.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
            
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.model_path = os.path.join(opt.ckpt_dir, opt.date, opt.ckpt_name)
        opt.result_path = os.path.join(opt.results_dir, opt.date, opt.ckpt_name)
        util.mkdir(opt.result_path)
        self.opt = opt
        return self.opt
