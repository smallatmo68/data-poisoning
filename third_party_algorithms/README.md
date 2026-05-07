# third_party_algorithms/

本目录用于存放第三方算法源码（只读参考，不直接 import）。

## 目录结构

```
third_party_algorithms/
├── alibi-detect-master/        ← Alibi Detect 库源码（分布漂移检测）
├── BackdoorBench-main/         ← BackdoorBench 后门基准框架
├── BackdoorDetection-main/     ← LLM 后门检测 + 内置数据集
├── cleanlab-master/            ← Cleanlab 标签投毒检测
├── defend_framework-main/      ← BagFlip/DPA 防御框架
├── DetectionFramework/         ← 废弃原型（不使用）
└── Influence-Based-Glitch-Detection-main/  ← 影响函数异常检测
```

## 使用说明

- **所有算法通过 `algorithm_engine/detectors/` 中的 adapter 调用**
- 不要直接在 Django views/services 中 import 这些目录
- 如需使用算法，通过 `algorithm_engine.registry.get_detector('detector_name')` 获取

## 当前状态

| 目录 | 状态 | 对应 Adapter |
|------|------|------|
| cleanlab-master/ | 已通过 pip 安装 cleanlab | `detectors/label_poison/cleanlab_detector.py` |
| alibi-detect-master/ | 可选安装 | `detectors/distribution/mmd_detector.py` |
| BackdoorBench-main/ | 参考实现 | `detectors/backdoor/spectral_signature_detector.py` |
| BackdoorDetection-main/ | 数据集来源 | `DATA_SOURCE_PATH` 配置 |
| defend_framework-main/ | 参考实现 | 待接入 |
| Influence-Based-Glitch-Detection-main/ | 参考实现 | `detectors/influence/influence_detector.py` |
| DetectionFramework/ | 废弃 | 不使用 |

## 注意

- 这些目录中的算法源码在根目录（如 `BackdoorBench-main/`）中也存在
- 建议将根目录的源码目录移动到此处（运行 init_system.py 时会自动处理）
- 实际检测使用的是 algorithm_engine 中的适配器，不直接使用此处的源码
