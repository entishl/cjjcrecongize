# 项目改造计划：部署到 Hugging Face Spaces

## 总体思路

我们将创建一个新的 `app.py` 文件，作为 Hugging Face Space 的入口。这个文件会使用 Gradio 库来构建一个 Web 用户界面。我们会重构 `main.py` 中的核心代码，使其能够被 `app.py` 调用，从而将命令行工具的功能无缝迁移到网页应用上。

## 改造计划详情

#### **第一步：环境与结构准备**

1.  **创建 Gradio 应用入口**:
    *   新建一个文件 `app.py`。这将是我们的 Gradio 应用主文件。

2.  **更新依赖**:
    *   在 `requirements.txt` 文件中添加 `gradio`，以确保部署时能正确安装。

3.  **配置 Hugging Face Space**:
    *   创建一个 `README.md` 文件（如果已有则修改）。Hugging Face Spaces 使用这个文件来配置应用。我们需要在文件顶部添加一段 YAML 配置，指定应用的标题、SDK（Gradio）、主文件等信息。

4.  **处理大数据文件**:
    *   像 `data/feature_database.npy` 这样的模型数据文件可能很大，需要使用 Git LFS (Large File Storage) 进行跟踪，以便能上传到 Hugging Face Hub。

#### **第二步：核心代码重构**

1.  **逻辑封装**:
    *   我们将 `main.py` 中的核心识别流程——从读取图片到返回结果——封装成一个独立的函数，例如 `recognize_and_visualize(image_array)`。
    *   这个函数将接收一个图片作为输入（由 Gradio 提供），并返回两个结果：一张标注了识别结果的可视化图片和一个包含详细数据的 Pandas DataFrame。

2.  **模型一次性加载**:
    *   `AvatarRecognizer` 类在初始化时需要加载模型和特征数据，这是一个耗时操作。我们会将其设置为一个全局变量，在应用启动时只加载一次，避免每次用户上传图片都重复加载。

3.  **功能函数适配**:
    *   修改 `main.py` 中的 `save_results_to_csv`、`visualize_results` 和 `save_failed_crops` 等辅助函数。它们不再直接将文件保存到磁盘，而是将处理好的数据（如图片对象、DataFrame）返回给主调用函数。
    *   对于“保存识别失败的切片”功能，我们会保留 `save_failed_crops` 的核心逻辑，但会将其输出目录指向 Hugging Face Space 服务器上的一个临时文件夹，方便后续下载和分析。

#### **第三步：构建 Gradio 界面**

在 `app.py` 中，我们将使用 Gradio 来实现您期望的交互界面：

1.  **输入组件**: 一个图片上传框，让用户可以拖拽或点击上传游戏截图。
2.  **输出组件**:
    *   一个图片显示框，用于展示返回的、标注好的结果图。
    *   一个数据表格 (DataFrame) 组件，用于清晰地展示 CSV 格式的详细识别结果。
3.  **启动应用**: 将重构后的核心处理函数与 Gradio 的输入/输出组件连接起来，并启动 Web 服务。

### 流程图

```mermaid
graph TD
    subgraph "Hugging Face Space (Gradio App)"
        A[用户上传截图] --> B{app.py};
        B --> C[调用核心识别函数];
    end

    subgraph "核心识别逻辑 (Refactored from main.py)"
        C --> D[1. 全局加载 Recognizer 模型];
        D --> E[2. 识别截图中的所有角色];
        E --> F[3. 生成可视化结果图];
        E --> G[4. 生成 CSV 数据];
        E --> H{有无法识别的切片?};
        H -- 是 --> I[5. 保存失败的切片到服务器];
        H -- 否 --> J;
    end

    subgraph "Hugging Face Space (Gradio App)"
        F --> K[在网页上显示结果图];
        G --> L[在网页上显示数据表格];
        I --> J[处理完成];
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px