import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    # 判断本地是否已经存有下采样factors或者对应分辨率的图像，如果没有需要重新加载
    # 抽取要求的下采样因子的图片
    # 如果不存在已经给定的下采样图片，就按照***段执行
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    # 抽取要求的分辨率的图片
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    # 无需新建
    if not needtoload:
        return
    
    
    # ***
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))  # 能够读取到[ImageSize, 位姿+boundaries] = [20, 17] 
                                                                    # 代表测试集中共有20张照片，每张照片有一个17维度的信息
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 抽取前15个特征，然后转变位置 原：[20, 15] ==> [20, 3, 5] ==> [3, 5, 20]
    bds = poses_arr[:, -2:].transpose([1,0])  # 抽出后两个特征，前置 [20, 2] ==> [2, 20]
                                              # 后两个特征就是最近和最远采样点作为boundaries
    
    # 抽取images中，所有的文件中以 jpg/png结尾的图片，放置入列表，并抽取第一张
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # 获取图像shape [h,w,c] = [3024, 4032,3]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    # 如果有相关的下采样因子，就对图像进行下采样
    # 给定缩放比/目标高/目标宽中任何一个都可以按照目标对输入图像进行下采样
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    # 输入了高宽之后，手动计算factor
    
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    # 判断是否存在采样后的路径
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    # 判断pose数量是否和图像一致
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # 获取图像shape
    # 产生的新的维度为[378, 504] == [3024/8,4032/8]
    sh = imageio.imread(imgfiles[0]).shape
    # poses.shape=[3,5,20] 
    # 现在抽取了ImageSize(对于所有图片)，的第一行和第二行的第四列，也就是第0行和第1行的第五列
    # 第五列存储的信息为[h(height),w(width),f(focallength)] 
    # 将原始图片的高宽(h,w)替换为更改分辨率/添加下采样后的图片高宽(更改分辨率)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # 将焦距f除以下采样因子
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    # 3x3是r旋转矩阵，3x1是t平移矩阵，3x1[h,w,f]
    
    # 默认加载图像，决定是否将图像文件返回
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f,format='PNG-PIL', ignoregamma=True)
        else:
            return imageio.imread(f)
    
    # 抽取所有图片的前三个维度，并在最后一个维度上拼接在一起
    # 构成imgs.shape=[378,504,3,20]    
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    # 它和模型训练没有关系，主要是用来生成一个相机轨迹用于新视角的合成
    # 知道这个函数它是想生成一段螺旋式的相机轨迹，相机绕着一个轴旋转，其中相机始终注视着一个焦点，相机的up轴保持不变
    # 首先是一个for循环，每一迭代生成一个新的相机位置。
    # c是当前迭代的相机在世界坐标系的位置 
    # np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])是焦点在世界坐标系的位置
    # z是相机z轴在世界坐标系的朝向
    # 接着使用介绍的viewmatrix(z, up, c)构造当前相机的矩阵
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def  recenter_poses(poses):
    # 将所有的pose做均值后逆转换，重新定义了世界坐标系，原点大致在被测物中心
    """
    第一步先用刚刚介绍的poses_avg(poses)得到多个输入相机的平均位姿c2w
    接着用这个平均位姿c2w的逆左乘到输入的相机位姿上就完成了归一化
    
    首先我们要知道利用同一个旋转平移变换矩阵左乘所有的相机位姿是对所有的相机位姿做一个全局的旋转平移变换
    那下一个问题就是这些相机会被变到什么样的一个位置
    我们可以用平均相机位姿作为支点理解
    如果把平均位姿的逆c2w^-1左乘平均相机位姿c2w
    返回的相机位姿中旋转矩阵为单位矩阵
    平移量为零向量。也就是变换后的平均相机位姿的位置处在世界坐标系的原点
    XYZ轴朝向和世界坐标系的向一致。
    """
    # pose=[20,3,5]
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    # 统合所有的相机角度，转换成一个20合一的平均位姿c2w
    c2w = poses_avg(poses)
    # [3,5] ==> [4,4 ]
    # 由于要对位姿做逆运算左乘，所以必须将他扩展成一个方阵
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    # 沿着poses的imageSize维度进行复制以便可以和所有的image进行维度拼接
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    # poses被扩展为[20,4,4]
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    # 取逆再左乘Pose
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    # pose变成了单位阵
    return poses


#####################
# 用来生成360°场景
"""
"360°场景"通常指的是渲染全景图像或全景视频的场景。这种场景涉及到相机在360度水平方向上的全方位视野,可以捕捉整个环境的信息。
对于一个360°场景,相机可以在水平方向上以任意角度旋转,从而捕捉整个环境的景象。这种场景可能包括室外景观、城市全景、室内全景等。
渲染这样的场景需要能够处理相机在水平方向上的自由旋转,以便在不同方向上生成全景图像
"""

def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    
    # pose=[3,5,20]
    # bds=[2,20]
    # imgs=[378,504,5,20]
    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    # ？可能因因为Colmap，llff转变的时候做的处理，[x,y,z] ==> [y,-x,z]？？？
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # 将最后一根轴移动到第一根
    # 将img_size移动到第一根轴
    # poses原来的shape [3,5,20] ==> [20,3,5]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # imgs原来的shape [378,504,3,20] ==> [20,378,504,3]
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    # bds原来的shape [2,20] ==> [20,2] 
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    """
    两个参数用于表示场景的范围Bounds (bds),是该相机视角下场景点离相机中心最近(near)和最远(far)的距离,所以near/far肯定是大于0的

    这两个值是怎么得到的？
    是在imgs2poses.py中,计算colmap重建的3D稀疏点在各个相机视角下最近和最远的距离得到的。
    这两个值有什么用？
    之前提到体素渲染需要在一条射线上采样3D点,这就需要一个采样区间,而near和far就是定义了采样区间的最近点和最远点。
    贴近场景边界的near/far可以使采样点分布更加密集,从而有效地提升收敛速度和渲染质量。
    """
    
    # ?coarse和fine的层级渲染，调整采样点分布集中在平面附近来优化，实现方式就是通过改变采样点的near和far？
    # Rescale if bd_factor is provided
    # 进行边界缩放的比例
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    # pose中的t和boundaries也要进行缩放
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        # 重定义世界坐标系
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 为render准备参数，见render_path_spiral()
        # pose的中心变成了单位阵，旋转矩阵为单位阵,平移矩阵为0
        # shape=(3,5)相当于汇集了所有图像
        c2w = poses_avg(poses)  # 计算所有相机位姿的平均位姿
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        # 最小深度和最大深度只是采样的最小和最大深度，并不是全局的最小深度和最大深度
        # 为了创造全景我们需要留出一些余量，以获得最好的效果
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        # 焦距
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        # 获取所有poses的3列，shape(图片数,3)
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        # 求90百分位的值
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        # 一个list，有120（由N_views决定）个元素，每个元素shape(3,5)
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    # shape 图片数
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    # 取到值最小的索引
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    
    # images (图片数,高,宽,3通道), poses (图片数,3通道,5) ,bds (图片数,2) render_poses(N_views,图片数,5)，i_test为一个索引数字
    return images, poses, bds, render_poses, i_test



