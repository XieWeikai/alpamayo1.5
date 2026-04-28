# 尝试编写alpamayo1.5的训练代码

当前仓库并没有alpamayo1.5的训练代码，我希望你帮我实现一个训练代码，使用accelerate这个包做训练，代码越简洁明了越好（可以是手写的训练循环），我只是希望能够验证可以训练起来这个模型。使用4卡做分布式训练，使用fsdp。你只需要用dummy dataset+dataloader即可，不需要用哪里来的真实数据进行训练。训练两部分内容，一个是VLM部分给定输入要有一块输出部分，用Next token prediction loss来学习输出部分(包括CoT和trajectory token)。另一部分是action expert，通过flow matching的方式来学习生成轨迹。要求action expert的attention mask可以比较容易的自定义，比如可以方便的在能看到VLM全部kv cache和能看到VLM的输入部分（看不到输出部分）这两种模式下切换。然后对于flow matching的训练上也有些技巧，即采样时我希望不要是均匀的从t在0-1中采样，而是如某篇论文中描述的那样，如下：
```latex
\paragraph{Sampling the flow matching timestep.}
The original flow matching papers~\cite{28,32} sample the flow matching timestep from a uniform distribution:
$\tau \sim \mathcal{U}(0,1)$.
Esser et al.~\cite{14} instead propose sampling from a logit-normal distribution that emphasizes the middle timesteps; the authors posit that at high timesteps (low noise levels), the model needs only to learn the identity function, and at low timesteps (high noise levels), the model needs only to learn the mean of the data distribution.
However, we hypothesize that the task of action prediction is subtly different from high-resolution image synthesis --- while it may be relatively easy to predict the mean image conditioned on a text label, predicting the mean action conditioned on a robot observation (i.e., learning $\mathbb{E}[A_t \mid o_t]$) is a much harder problem; this is because the observation $o_t$ is very informative in that it should constrain the distribution of possible actions much more than a text label constrains the distribution of possible images.
As a result, we designed a timestep sampling distribution that emphasizes low timesteps (high noise levels); additionally, timesteps above a given threshold $s$ are not sampled at all, since they are not needed so long as the integration step $\delta$ is greater than $1-s$.
The distribution is given by
\[
p(\tau) = \mathrm{Beta}\!\left(\frac{s-\tau}{s};\,1.5,\,1\right)
\]
and is visualized in Figure~14.
We use $s = 0.999$ in our experiments, which allows for $\delta > \frac{1}{1000}$, or up to 1{,}000 integration steps.
```


## 要求

你完成前面描述的任务我希望能够做到尽量不要修改已有的代码，不要破坏这个仓库已有的内容，而是独立的增加内容来完成我说的这个任务。

你写的代码要简洁明了，各个部分如训练循环、模型、数据、loss不要太过耦合，而应该比较模块化容易分开来修改各个部分。

你进行的每一步每一个里程碑都要有一个新文档记录，放在progress目录内（你自己创建吧），每一次里程碑式的修改都要有独立的agent进行review，并有独立的agent设计测试并进行测试。如何设计测试和测试结果也要落成文档，和milestone一一对应，放在test目录内。

## 一些已有的信息
当前仓库的mamba(快速版conda)环境在/data/envs/alpamayo，你直接使用这个环境，缺少的包比如accelerate也是装在这里面即可。
模型在/data-25T/models/Alpamayo-1.5-10B这里面，你可以从这里load模型进行初始化。

