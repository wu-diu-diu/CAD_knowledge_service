# PPO 模型结构说明

## 1. 总体结构

本项目在 `RL/model.py` 中实现的是一个 **共享编码器 + 双解码器** 的 PPO Actor-Critic 模型，用于房间内灯具布局。

整体数据流为：

```text
观测状态 obs
-> SharedEncoder
-> PolicyDecoder  -> 动作 logits
-> ValueDecoder   -> 状态价值 V(s)
```

其中：

- `PolicyDecoder` 对应 PPO 中的 **Actor**
- `ValueDecoder` 对应 PPO 中的 **Critic**
- `SharedEncoder` 负责提取公共空间特征，减少重复计算

---

## 2. 输入状态

模型输入张量形状为：

```text
[B, 4, 32, 32]
```

4 个通道分别为：

1. `room_mask`：房间内部区域
2. `placed_lamps`：当前已经放置的灯具
3. `switch_mask`：开关位置
4. `door_mask`：门的位置

这里 `B` 为 batch size，空间尺寸固定为 `32 × 32`。

---

## 3. 共享编码器 SharedEncoder

共享编码器采用 U-Net 风格的下采样编码结构，共有 4 个阶段：

```text
stage1: [B,  32, 32, 32]
stage2: [B,  64, 16, 16]
stage3: [B, 128,  8,  8]
stage4: [B, 256,  4,  4]
```

每个 `ConvBlock` 包含两层卷积：

- `Conv2d`
- `GroupNorm`
- `ReLU`

其中 `stage2`、`stage3`、`stage4` 之间通过 `MaxPool2d(2)` 下采样。

### 编码器作用

- 提取房间几何结构特征
- 编码当前灯具布局状态
- 为策略头提供空间细节和高层语义
- 为价值头提供全局状态表示

---

## 4. 策略解码器 PolicyDecoder

策略解码器是一个 U-Net 风格上采样结构：

```text
stage4 -> up3 + skip(stage3)
      -> up2 + skip(stage2)
      -> up1 + skip(stage1)
      -> spatial_head
```

### 4.1 空间动作分支

经过 3 次上采样后，特征恢复到 `32 × 32`，再通过 `1×1 Conv` 得到：

```text
spatial_logits: [B, 1024]
```

对应 `32 × 32` 网格中每一个位置的放灯动作。

### 4.2 停止动作分支

同时，从 bottleneck 特征 `stage4` 上通过：

- `AdaptiveAvgPool2d(1)`
- `Linear(256 -> 128)`
- `ReLU`
- `Linear(128 -> 1)`

生成一个：

```text
stop_logit: [B, 1]
```

表示“停止布局”动作。

### 4.3 最终动作空间

最终动作 logits 为：

```text
[B, 1025]
```

含义为：

- `0 ~ 1023`：在某个网格位置放灯
- `1024`：停止动作

---

## 5. 动作掩码机制

模型在策略头中直接做了动作合法性掩码。

### 5.1 放灯动作掩码

只有满足以下条件的位置才允许放灯：

- 在房间内部
- 不是门
- 不是开关
- 不是已经放过灯的位置

非法位置的 logit 会被置为：

```text
NEG_INF = -1e9
```

这样在 `Categorical(logits=...)` 中其概率会近似为 0。

### 5.2 停止动作掩码

如果设置了 `target_lamp_count`，那么在当前灯数未达到目标前：

- `stop` 动作也会被 mask 掉

只有达到目标灯数后，停止动作才合法。

这保证了贪婪推理和训练采样都不会提前停止。

---

## 6. 价值解码器 ValueDecoder

价值网络不恢复空间分辨率，而是直接做多尺度全局池化。

它从编码器的三个尺度提取全局信息：

- `stage2 -> 64 维`
- `stage3 -> 128 维`
- `stage4 -> 256 维`

拼接后得到：

```text
fused: [B, 448]
```

再经过 MLP：

```text
448 -> 256 -> 128 -> 1
```

输出：

```text
value: [B, 1]
```

表示状态价值 `V(s)`。

### 设计原因

这种设计比只用最深层特征更稳定，因为：

- `stage2` 保留较多局部结构
- `stage3` 保留中尺度布局信息
- `stage4` 保留全局语义

---

## 7. PPO 中的动作采样与评估

### 7.1 `act()`

`act()` 用于 rollout 阶段。

流程为：

1. 前向计算 `logits` 和 `value`
2. 构造 `Categorical(logits=logits)`
3. 根据模式选择动作

- `deterministic=False`：采样动作 `dist.sample()`
- `deterministic=True`：贪婪动作 `argmax(logits)`

输出包括：

- `action`
- `log_prob`
- `value`
- `logits`

其中：

- `action` 是动作索引
- `log_prob = log π(a|s)`，是 PPO 计算策略比值的重要量

### 7.2 `evaluate_actions()`

`evaluate_actions()` 用于 PPO 更新阶段。

它对一批 `(obs, actions)` 重新计算：

- `log_prob`
- `entropy`
- `value`

用于构造 PPO 损失：

- 策略损失
- 值函数损失
- 熵正则项

---

## 8. 模型对应的 PPO 角色

从强化学习角度看：

- `SharedEncoder`：公共状态表示提取器
- `PolicyDecoder`：策略函数 `π(a|s)`
- `ValueDecoder`：状态价值函数 `V(s)`

所以该模型是一个典型的：

```text
共享 backbone 的 Actor-Critic PPO 网络
```

---

## 9. 当前模型特点

当前实现有几个明显特点：

1. **显式空间建模**
   - 输入是二维房间网格
   - Actor 直接输出空间动作分布

2. **共享特征提取**
   - Actor 和 Critic 共用编码器
   - 提高样本效率

3. **动作合法性内置**
   - 非法格点不会被采样
   - 未达目标灯数前不能提前停止

4. **空间策略 + 全局价值**
   - 策略头保留空间细节
   - 价值头更强调全局布局质量

---

## 10. 总结

本项目的 PPO 模型本质上是一个面向 `32×32` 房间网格的 U-Net 风格 Actor-Critic 网络。共享编码器负责提取房间与当前布局的空间特征；策略解码器输出每个网格位置的放灯动作及停止动作；价值解码器输出当前状态价值。通过动作掩码机制，模型能够在策略层面直接排除非法放灯位置，并约束停止动作的触发时机，从而更稳定地服务于房间灯具布局任务。
