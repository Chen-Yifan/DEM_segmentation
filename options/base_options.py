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
        parser.add_argument("--isTrain", type=bool, default=1, help='is train or test')
        parser.add_argument("--model", type=str, default='unet', help='choose which model to use')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='/home/yifanc3/models/', help='models are saved here')
        parser.add_argument('--augmentation', type=int, default=0, help='do augmentation or not')
        parser.add_argument("--results_dir", type=str, default='/home/yifanc3/results/', help='results are saved here')
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--optimizer", type=int, default=1, help='1:Adam, 2:Adadelta')
        parser.add_argument('--input_channel', type=int, default=5)
        parser.add_argument('--input_shape', type=int, default=128, help='input image width/height')
        parser.add_argument('--k', type=int, default=1, help='k_fold cross validation')
         
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
        expr_dir = os.path.join(opt.checkpoints_dir, opt.date, opt.name)
        util.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
            
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.Model_path = os.path.join(opt.checkpoints_dir, opt.date, opt.name)
        opt.Result_path = os.path.join(opt.results_dir, opt.date, opt.name)

        self.opt = opt
        return self.opt
