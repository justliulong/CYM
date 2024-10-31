## 自定义模型
- 在文件` ultralytics/models/yolo`目录下创建一个目录`cym`用于存储自定义模型需要使用的训练器、预测器、验证器（这些可以沿用`yolo`模型的项目），同时在`cym`目录下记得创建`__init__.py`文件以便导出（其他文件是可以使用的），内容也参考沿用`yolo`模型的项目

- 在`ultralytics/models/yolo/model.py`文件中自定义模型，模仿之前的`yolo`和`yolo-world`定义，之后需要在`ultralytics/models/yolo/__init__.py`文件夹下导出你的自定义模块`CYM`：

  ```py
  # ultralytics/models/yolo/__init__.py
  from ultralytics.models.yolo import classify, detect, segment, world, cym
  
  from .model import YOLO, YOLOWorld, CYM
  
  __all__ = "classify", "segment", "detect", "world", "YOLO", "YOLOWorld", "cym"
  
  #! 上面小写的是目录，下面大写的是模型！
  ```

  > ps：这个模型的名称一定不能和第一步用于存储自定义模型需要使用的训练器、预测器、验证器的目录名相同，允许一个大写一个小写（尽量是文件夹名称小写，而模型的名称大写，这样规范）

- 定义模型结构，在`ultralytics/cfg/models/`目录下，创建一个属于你模型`cym`目录，其中存放模型结构的`yaml`文件，模型结构具体也参考yolo文件

- 如果训练器、验证器等都沿用`yolo`模型的，那接下来调用就和`yolo`一样比较方便：

  ```python
  # train.py
  from ultralytics import CYM
  
  model = CYM("ultralytics/cfg/models/cym/cym.yaml")、
  
  result = model.train(data='ultralytics/cfg/datasets/bcc.yaml',
                       epochs=100,
                       imgsz=640,
                       batch=4,
                       workers=1,
                       project='runs/train',
                       name='cym-bcc')
  ```
  
## 添加自定义模块

- 自定义模块的定义文件可以放置在`ultralytics/nn/modules`目录下，可以在此目录下建立文件并且自定义模块：

  ```python
  # ultralytics/nn/modules/Mamba/custom.py
  
  from torch import nn
  
  class VSSM(nn.Module):
      """Virtual Spatial Sampling Module (VSSM)"""
  
      def __init__(self, c1, c2):
          super().__init__()
          print("you successful used VSSM!!! the input channel is {}, the output channel is {}".format(c1, c2))
  
      def forward(self, x):
          return x
  ```

- 导出模型需要修改`ultralytics/nn/modules/__init__.py`文件来导出自定义模块：

  ```python
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  
  from .block import (
      ...
  )
  from .conv import (
      ...
  )
  from .head import ...
  from .transformer import (
      ...
  )
  
  from .Mamba.custom import (VSSM)
  
  __all__ = (
      ...
      "VSSM",
  )
  
  ```

- 最后还需要修改`ultralytics/nn/tasks.py`文件下的**`parse_model`**函数来解析模块：

  ```python
  # task.py
  
  ...
  from ultralytics.nn.modules import (
      ...
      VSSM,
  )
  ...
  
  def parse_model(...):
      ...
      for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args 在这里添加模快解析
         ...
         elif m is VSSM:
              # c1 是模块输入通道数 == 上一层的模块输出
              # c2 是模块的输出通道数 == 下一层的模块输入
              # ch 记录每一层的输出通道数
              # args 是模块初始化的构造
              # 一般需要确认c2的数值
              # ch[-1]代表着上一个模块的输出通道。args[0]是默认的输出通道
              c2 = ch[f]
  ```

  

- 最后在`ultralytics/cfg/models/cym/cym.yaml`中就可以使用这个模块了：

  ```yaml
  ...
    - [-1, 1, VSSM, [1024, 1024]]
  ...
  ```




[issue](https://kkgithub.com/ultralytics/ultralytics/issues/7312)中提到了这点，所以我将这个预先训练的检查点放置在这个地方