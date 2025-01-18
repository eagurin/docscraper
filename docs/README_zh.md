# DocScraper

一个异步文档抓取和解析器，可以爬取网站、提取内容并生成结构化的markdown文档。

[English](README_en.md) | [Русский](README_ru.md)

## 功能特点

- 异步网页爬取与并发处理
- 按域名组织文档
- 使用OpenAI辅助生成Markdown
- 结构化输出与合并文档
- 多级别的完整日志系统
- Docker支持与资源管理

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/docscraper.git
cd docscraper
```

2. 环境设置：
```bash
cp .env.example .env  # 编辑配置
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. 使用Docker运行：
```bash
docker-compose up --build
```

或本地运行：
```bash
python main.py
```

## 配置

### 环境变量 (.env)
- `MODEL_NAME`：OpenAI模型 (默认: gpt-4)
- `OPENAI_API_KEY`：您的OpenAI API密钥

### 日志系统
- 控制台 (INFO级别)：彩色输出，实时更新
- 文件 (DEBUG级别)：详细诊断信息保存在 `logs/docparser_{time}.log`

## 输出结构

```
docs_output/
├── sites/              # 按域名组织的文档
│   └── {domain}/      # 每个域名的内容
└── combined/          # 汇总文档
	├── index.md       # 主索引
	└── {domain}.md    # 域名汇总
```

## Docker支持

- 资源限制：2GB内存
- 文档和日志的卷挂载
- 启用健康检查

## 许可证

MIT License - 详见 [LICENSE](LICENSE)