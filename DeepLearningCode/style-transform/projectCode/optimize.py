from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

#定义好图像生成和VGG网络。
#网络生成模型是实际生成东西和模型的，vgg主要是特征提取。
#

#论文里是四层，但因为调用的vgg原因，我们这里style用五层。
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#content有一层
CONTENT_LAYER = 'relu4_2'
#cpu
DEVICES = ''

#优化函数
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1
    #首先考虑batchsize是否有多余。多余的需要去掉，用[:-mod]表示。
    mod = len(content_targets) % batch_size
    if mod > 0:
        # 去掉余数，如果是-1就是包含所有，反过来的形式，所以这样写就是取余数之前所有元素
        content_targets = content_targets[:-mod]
    #我们优化的是多个图的特征图的特征,所以用字典表示，每个图中各个特征放入里面。
    #记住我们的核心就是优化每个层之间生成的特征图的特征。然后比较各个特征图之间的差异值。所以放入一起。
    style_features = {}
    #彩色图，记住用元组格式写。
    batch_shape = (batch_size,256,256,3)
    #style的batch肯定是1个。所以数量是1，然后再加上自身shape(高,宽,通道数)即可。
    style_shape = (1,) + style_target.shape

    #print(style_shape)

    #计算style层
    #先拿出Graph计算图,再指定拿什么去训练即device，然后是session
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        #先开辟空间
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        #直接用vgg预处理。
        style_image_pre = vgg.preprocess(style_image)
        #调用vgg进行网络操作。
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target]) #别忘了转化np格式,这个是stype的特征图
        #处理完VGG,开始循环操作,将style的各个层设置，获取各个层的特征图。
        for layer in STYLE_LAYERS:
            #将style特征图放入feature里。
            features = net[layer].eval(feed_dict={style_image:style_pre})
            #我们的本质是比较各个特征图之间的差异值。所以关心的是整体的特征图。
            #第一个维度:特征图的所有信息，所以设置-1，第二个维度:特征图数量，取最后一个维度。
            #而我们正常是特征图高，宽，输入数(通道数)，输出数(个数)。所以只取前三个并一起成-1，最后一个单独取出来。
            features = np.reshape(features, (-1, features.shape[3]))
            #print (features.shape)
            #矩阵的转置*矩阵自身。因为我们要看的是每个特征图与特征图之间的关系。论文叫这个是gamer值。而这个gamer值即使特征
            gram = np.matmul(features.T, features) / features.size
            #将特征赋值。
            style_features[layer] = gram

    #计算原图层-content
    with tf.Graph().as_default(), tf.Session() as sess:
        #下面操作仿照style
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        #下面这个经过vgg的content，也就是没有经过transform生成网络的。
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        #原始图像在vgg初始化后，开始进入左边部分，也就是卷积+transform+反卷积
        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            #开始卷积+transform+反卷积，这段也就是生成网络，生成网络主要是原始图像处理。
            preds = transform.net(X_content/255.0) #归一化。
            #经过transform生成网络后，开始vgg网络最终求得loss值。
            #下面是预处理。
            preds_pre = vgg.preprocess(preds)

        #下面我们要进行测试，一个是经过生成网路也就是transform操作，一个是没经过。
        #这段是经过生成网络，直接进行vgg操作。
        net = vgg.net(vgg_path, preds_pre)
        #当前特征图整体大小，将上面没经历生成网络的传进来。再乘以batch_size
        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        #权重项。通过l2正则化。用经过生成网络 net[CONTENT_LAYER]与没有经历生成网络content_features[CONTENT_LAYER]的相减，
        #这样计算出俩者的差值，我们希望这个差值越小越好，这样我们就不用再跑vgg网络，直接跑生成网络一个正向传播即可。
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        #下面计算style_loss值。
        style_losses = []
        for style_layer in STYLE_LAYERS:
            #获取层数
            layer = net[style_layer]
            #参数
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            #图大小，每次卷积相当于一张图*数量，所以是filter数量*长宽就是图大小。
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            #转置操作,我们最后是计算gama值。
            feats_T = tf.transpose(feats, perm=[0,2,1])
            #根据论文，gama就是矩阵自己与转置相乘。这个gama值就是我们最后计算的她特征值。
            grams = tf.matmul(feats_T, feats) / size
            #以上就是在生成网络上算出的gama值。
            #下面这个gama值是不用生成网络生成的结果。
            style_gram = style_features[style_layer]
            #然后用俩种方式跑出的gama值相减，越小越好。
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        #最后汇总得出最终的style 的 loss值，这个越小越好。
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        #最终Loss，再加个降噪loss值tv_loss
        loss = content_loss + style_loss + tv_loss

        #网络训练。用AdamOptimizer
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        import random

        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)  #样本数
            iterations = 0
            #迭代次数*batch小于样本数即可。
            while iterations * batch_size < num_examples:
                start_time = time.time()
                #当前batch，也就是从哪开始的batch
                curr = iterations * batch_size
                #当前batch后的结果，到时候就取curr~step之间
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                    #读取数据到样本中。
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }
                #开始跑
                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                #下面是打印操作。
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }
                    #最终计算。
                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                        #保存模型
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    #返回的值。
                    yield(_preds, losses, iterations, epoch)

#求当前层的特征图大小，所以用reduce聚合。
def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
