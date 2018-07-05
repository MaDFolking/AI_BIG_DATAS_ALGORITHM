import sys,os,pdb
#sys.path.insert(0,src)  #加载src文件夹里的python文件
import numpy as np,scipy.misc   #scipy是加载图像
from scipy import optimize  #优化
from argparse import ArgumentParser  #命令行解析
from utils import save_img,get_img,exists,list_files
import evaluate

CONTENT_WEIGHT = 7.5e0  #原图权重
STYLE_WEIGHT = 1e2      #style图权重
TV_WEIGHT = 2e2         #全变差损失权重，为了让图像看的更真实一些

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2   #因为图片数量比较多，所以我们设置迭代次数小一些。
CHECK_POINTS_DIR = "D:\\spark\\transform-style\\checkpoints\\"   #保存路径
CHECK_POINTS_ITERATIONS = 2000                                  #保存路径的迭代次数
VGG_PATH = "D:\\spark\\transform-style\\datas\\imagenet-vgg-verydeep-19.mat"   #VGG模型路径,我们还需要从这个.mat结构读取出权重值
TRAIN_PATH = "D:\\spark\\transform-style\\data-model"  #训练文件夹,就是训练的图片
BATCH_SIZE = 4  #每次迭代多少张
DEVICE = 'cpu:0'

#命令行解析
def build_parser():
    parser = ArgumentParser()  #创建ArgumentParser对象
    #指定路径
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)
    #--是指定函数参数
    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECK_POINTS_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

#检查
#我们的命令是否正确，是否出问题。中间文件夹是否存在等等。
def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def main():
    parser = build_parser()
    options = parser.parse_args()  #上面构建完就需要解析
    check_opts(options)            #检查

    #获取图片
    style_target = get_img(options.style)
    #判断是否为训练和测试
    if not options.slow:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]

    kwargs = {
        "slow": options.slow,
        "epochs": options.epochs,
        "print_iterations": options.checkpoint_iterations,
        "batch_size": options.batch_size,
        "save_path": os.path.join(options.checkpoint_dir, 'fns.ckpt'),
        "learning_rate": options.learning_rate
    }

    if options.slow:
        if options.epochs < 10:
            kwargs['epochs'] = 1000
        if options.learning_rate < 1:
            kwargs['learning_rate'] = 1e1

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)
        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir, epoch, i)
            if not options.slow:
                ckpt_dir = os.path.dirname(options.checkpoint_dir)
                evaluate.ffwd_to_img(options.test, preds_path,
                                     options.checkpoint_dir)
            else:
                save_img(preds_path, img)
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)


if __name__ == '__main__':
    main()





































