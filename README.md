---
title: NIKKE Arena CV
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.48.0
app_file: app.py
pinned: false
---

# NIKKE 竞技场角色识别器

这是一个使用 Gradio 构建的 Hugging Face Spaces 应用，用于识别游戏《胜利女神：妮姬》竞技场截图中的角色。

## 如何使用

1.  点击或拖拽上传一张游戏竞技场界面的截图。
2.  点击“开始识别”按钮。
3.  稍等片刻，右侧将显示标注了识别结果的图片和详细的角色阵容表格。

## 功能

-   自动识别截图中双方队伍的5v5角色。
-   可视化展示识别结果。
-   提供表格化的详细数据。
-   自动保存在识别中置信度较低的头像，以便于后续优化模型。