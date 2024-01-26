# MHIM

论文：Multi-grained Hypergraph Interest Modeling for Conversational Recommendation
原文地址: [[arXiv]](https://arxiv.org/abs/2305.04798)


## Requirements

```
python==3.8.12
pytorch==1.10.1
dgl==0.4.3
cudatoolkit==10.2.89
torch-geometric==2.0.3
transformers==4.15.0
```

## Datasets

[Google Drive](https://drive.google.com/drive/folders/1witl2Ga8pQzAsreQhj4QUH7TldzWKzLa?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1WQoWOSrquIZtJz8AGfg9Cg?pwd=mhim)

将已处理的数据集下载下来, 解压 `data_contrast.zip` 文件并移动到 `Contrast/`文件夹, 解压 `data_mhim.zip` 文件并移动到 `MHIM/`文件夹。

创建python==3.8的虚拟环境
```powershell
conda create -n py38 python=3.8
conda activate py38 # 激活环境
```
下载所需库
```powershell
conda install loguru
conda install yaml
# 下载yaml后运行命令后报错，要下载pyyaml
conda install pyyaml
conda install transformers
conda install dgl
# 下载dgl时提示下载源不存在 换成pip下载，注意要写版本号，因为直接下载会导致与torch版本不兼容
pip install dgl==0.4.3
# 同理torch_geometric也遇到一样的情况
pip install torch_geometric==2.0.3
pip install torch-sparse==0.6.18
```
下载pytorch
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1699710956285-481d8447-f748-43d5-9347-7afffa8331da.png#averageHue=%23e8ead2&clientId=u9eb479ab-908f-4&from=paste&height=314&id=uef12ff0f&originHeight=528&originWidth=1394&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=67511&status=done&style=none&taskId=uac127704-82b1-44ac-9a56-dce5709a811&title=&width=829.7618717901304)
```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Quick Start

### Contrastive Pre-training

预训练R-GCN编码器:

```
cd Contrast
python run.py -d redial -g 0
python run.py -d tgredial -g 0
```

运行完成后将训练好的模型 `save/{dataset}/{#epoch}-epoch.pth` 文件移动到 `MHIM/pretrain/{dataset}/`.


### Running

运行时注意先修改`\MHIM-main\MHIM\crslab\data\dataset\hredial\hredial.py`和`\MHIM-main\MHIM\crslab\data\dataset\htgredial\htgredial.py`中的文件路径。
```
cd ../MHIM
python run_crslab.py --config config/crs/mhim/hredial.yaml -g 0 -s 1 -p -e 10
python run_crslab.py --config config/crs/mhim/htgredial.yaml -g 0 -s 1 -p -e 10
```

实验结果将会存放在 `MHIM/log/`文件夹中。

注意，运行python run_crslab.py --config config/crs/mhim/hredial.yaml -g 0 -s 1 -p -e 10时可能会报错：
![无法下载redial_nltk.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1698676966567-5ceffa00-0fb9-458d-81d2-46af76a3a1b9.png#averageHue=%23242221&clientId=u366ba15c-649f-4&from=paste&height=788&id=ub9fea27e&originHeight=1324&originWidth=2193&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=688475&status=done&style=none&taskId=uf3a84f42-4730-4ea0-9b82-ddd80515d9c&title=&width=1305.3570909869125)
发现代码中供下载的链接页面不存在了
![链接不存在.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1698676955258-2ec4234e-516a-481f-8dd6-af68c53c2419.png#averageHue=%23242120&clientId=u366ba15c-649f-4&from=paste&height=910&id=u65c61260&originHeight=1528&originWidth=2559&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=287625&status=done&style=none&taskId=uc5259e35-4b30-4371-8bb7-811c6ce5832&title=&width=1523.2142251871906)
在GitHub上直接搜索查找：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1698677063455-a5e7f1fe-a3a3-4a93-8730-473c2da1669c.png#averageHue=%23e4cfa2&clientId=u366ba15c-649f-4&from=paste&height=801&id=u30539f76&originHeight=1345&originWidth=2559&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=311285&status=done&style=none&taskId=uf53f0cfa-a536-4df7-bcf0-71a732b0dee&title=&width=1523.2142251871906)
下载：
[https://drive.google.com/drive/folders/1witl2Ga8pQzAsreQhj4QUH7TldzWKzLa](https://drive.google.com/drive/folders/1witl2Ga8pQzAsreQhj4QUH7TldzWKzLa)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1698716091387-81647991-d3c9-4a35-94c4-09654986352e.png#averageHue=%23e3ca89&clientId=u366ba15c-649f-4&from=paste&height=327&id=u8db3f726&originHeight=550&originWidth=2560&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=144028&status=done&style=none&taskId=u8e7a1cc8-7517-40c0-92b7-14e8768ecc3&title=&width=1523.809463258776)
将别人的代码中更新的链接打开后重新下载到指定位置：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1698677120883-27507a85-2216-4374-8d06-546cd14c94ff.png#averageHue=%2364635d&clientId=u366ba15c-649f-4&from=paste&height=49&id=u81846e55&originHeight=82&originWidth=1861&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=186195&status=done&style=none&taskId=u0397a0f0-58fc-47e1-beb3-ebbb098bf18&title=&width=1107.7380512205398)
包括hredial和htgredial
![image.png](https://cdn.nlark.com/yuque/0/2023/png/35653622/1698716250904-2fa9f77b-f315-40bf-a12f-2fe04effac23.png#averageHue=%23fdfdfc&clientId=u366ba15c-649f-4&from=paste&height=128&id=u5251b40a&originHeight=215&originWidth=1007&originalType=binary&ratio=1.6800000667572021&rotation=0&showTitle=false&size=33239&status=done&style=none&taskId=u85aa7d1e-edcc-4600-acea-210d324ba75&title=&width=599.4047380865576)
同时更改download.py中的build函数，当文件存在时不下载。

```python
def build(dpath, dfile, version=None):
    # print(dir(dfile))
    # print(dpath,dfile)
    if not check_build(dpath, version):
        logger.info('[Building data: ' + dpath + ']')
        if check_build(dpath):
            remove_dir(dpath)
        make_dir(dpath)

        # Check if the file already exists in the destination directory.
        file_path = os.path.join(dpath, dfile.file_name)  # 使用dfile对象的filename属性获取文件名
        if not os.path.exists(file_path):
            # Download the data if it doesn't exist.
            dfile.download_file(dpath)

        mark_done(dpath, version)
```

## Improvement

我对模型进行了两方面的改进：引入外部电影评论数据集进行数据增强和改变超边扩展的条件

### 引入外部电影评论数据集进行数据增强
引入外部电影评论数据集进行数据增强修改部分：
`\MHIM-main\MHIM\crslab\data\dataset\hredial\hredial.py`以及`\MHIM-main\MHIM\crslab\data\dataset\htgredial\htgredial.py`

### 改变超边扩展的条件
1. 修改扩展项的阈值
`\MHIM-main\MHIM\crslab\data\dataset\hredial\hredial.py`中的_search_extended_items函数（460~463行）
`\MHIM-main\MHIM\crslab\data\dataset\htgredial\htgredial.py`
中的_search_extended_items函数（258~261行）

2. 修改n跳邻居
`\MHIM-main\MHIM\crslab\model\crs\mhim\mhim.py`中的_build_adjacent_matrix函数（222行）

## Acknowledgement

这一实现基于开源的CRS工具包[CRSLab](https://github.com/RUCAIBox/CRSLab)。

如果您使用我们的代码或经过处理的数据集，请引用以下论文作为参考文献。
```
@inproceedings{shang2023mhim,
  author = {Chenzhan Shang and Yupeng Hou and Wayne Xin Zhao and Yaliang Li and Jing Zhang},
  title = {Multi-grained Hypergraph Interest Modeling for Conversational Recommendation},
  booktitle = {{arXiv preprint arXiv:2305.04798}},
  year = {2023}
}
```

