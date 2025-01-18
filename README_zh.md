# 🚀 DocScraper

> 将文档网站转换为针对AI训练优化的markdown文件集合

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🌍 **语言**: [English](README.md) | [Русский](README_ru.md) | [中文](README_zh.md)

## 🎯 这是什么

DocScraper 自动将文档网站转换为清晰、结构化的markdown文件，专为RAG（检索增强生成）系统和AI训练数据集优化。是创建高质量AI模型训练数据的理想工具。

### ✨ 核心功能

- 🔄 **智能爬虫**: 异步、多线程网站处理
- 📝 **智能转换**: HTML → 清晰Markdown
- 🧠 **AI增强**: OpenAI驱动的内容结构化
- 📊 **RAG优化**: 完美适配训练数据准备
- 🔍 **丰富元数据**: 保留上下文和关系
- 🐳 **Docker就绪**: 轻松部署和扩展

## 💫 为什么选择DocScraper？

- 📚 **清晰文档**: 完美格式化的markdown文件
- 🤖 **AI就绪格式**: 针对RAG系统优化
- 🌳 **结构保留**: 维持原始层次结构
- 🔗 **智能引用**: 保持内部链接和上下文
- 🎨 **丰富元数据**: AI生成的增强洞察

## 🚀 快速开始

### 前置要求

- 🐍 Python 3.8+
- 🔑 OpenAI API密钥

### 📦 安装

```bash
# 安装依赖
make install
```

**配置**:
```bash
cp .env.example .env
# 编辑 .env 设置:
# MODEL_NAME=gpt-4
# OPENAI_API_KEY=你的密钥
```

### 🎮 使用方法

```bash
# 运行爬虫
make run URL=https://docs.example.com
```

参数:
- URL: 要爬取的URL（必需）
- OUTPUT_DIR: 输出目录（默认：docs_output）
- MAX_CONCURRENT: 最大并发请求数（默认：3）
- WAIT_TIME: 页面加载等待时间（默认：10.0）
- MODEL: OpenAI模型（默认：gpt-4）


## 🛠 开发

### 设置开发环境
```bash
poetry install --with dev
poetry shell
```

### 代码质量
```bash
poetry run black .
poetry run isort .
poetry run mypy .
poetry run ruff .
```

### 测试
```bash
poetry run pytest
```

## 📁 项目结构

```plaintext
docscraper/
├── 📂 docs_output/        # 生成的文档
│   ├── sites/           # 按站点的内容
│   └── combined/        # 统一知识库
├── 📝 main.py           # 核心应用
├── 📄 Dockerfile        # 构建配置
├── 📋 requirements.txt  # 依赖项
└── ⚙️ .env             # 配置文件
```

## 🎨 输出格式

DocScraper生成两种RAG优化的内容：

### 1. 📑 站点特定文档
- 每页清晰的markdown
- 原始URL结构
- 丰富的元数据头
- AI增强的内容

### 2. 📚 统一知识库
- 交叉引用文档
- 全局搜索索引
- 主题关系
- 语义连接

## ⚙️ 配置

### 环境变量
| 变量 | 用途 | 默认值 |
|------|------|--------|
| MODEL_NAME | OpenAI模型 | gpt-4 |
| OPENAI_API_KEY | API认证 | 必需 |
| LOG_LEVEL | 日志详细程度 | INFO |
| MAX_CONCURRENT | 并行操作数 | 3 |

### 🔧 资源设置
- 🖥️ 内存限制: 2GB
- 📊 并发任务: 3
- 📝 日志轮转: 500MB
- 🕒 日志保留: 10天

## 📈 RAG集成

### 文档处理
- 📝 一致的markdown格式
- 🌳 层次结构
- 🏷️ 丰富元数据包含
- 🔍 语义分块
- 🔗 交叉引用

### 知识组织
- 📚 主题关系
- 🔄 文档依赖
- 🧩 语义连接

## 🤝 贡献

欢迎贡献！查看我们的[贡献指南](docs/CONTRIBUTING.md)了解：
- 📝 代码风格指南
- 🔍 测试要求
- 🚀 PR流程
- 📦 开发环境设置

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)