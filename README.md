# VQNet 2.18 API 文档（spain-chn 中文版）

基于 Sphinx 的 VQNet 2.18 API 与示例文档（中文翻译分支）。

仓库根目录的 `vqnet_spain-chn.pdf` 为一份已构建好的 PDF 快照（未纳入 git）。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 构建 HTML

```bash
make html
# 等价命令：
sphinx-build -b html source build/html
```

产物：`build/html/index.html`

如 PATH 中没有 `sphinx-build`，可指定解释器：

```bash
make html SPHINXBUILD=/path/to/sphinx-build
```

## 构建 PDF

```bash
make latexpdf
```

产物：`build/latex/vqnet.pdf`

若环境中 `latexmk`/`xindy` 异常，可用手动流程（与 `make latexpdf` 等价）：

```bash
sphinx-build -b latex source build/latex
cd build/latex
xelatex -interaction=nonstopmode vqnet.tex   # 第 1 遍：生成 aux/toc
xelatex -interaction=nonstopmode vqnet.tex   # 第 2 遍：目录、交叉引用
xelatex -interaction=nonstopmode vqnet.tex   # 第 3 遍：页码稳定
```

## PDF 字体与工具链说明

- TeX 引擎：`xelatex`（`conf.py` 中 `latex_engine` 已指定）。
- 中文字体：`conf.py` 已内置跨平台回退
  - Windows：SimSun / SimHei / Microsoft YaHei
  - Linux：Noto CJK SC（系统包 `fonts-noto-cjk`）
- 拉丁等宽字体（代码块/制表符线框）：Consolas（Windows）/ DejaVu Sans Mono（Linux）。
- **PDF 版字母索引已移除**（`conf.py` 中 `\let\printindex\relax`、`\let\makeindex\relax`）：
  当前 Sphinx 生成的 `sphinx.sty` 缺少 `\spxentry` 宏定义，索引条目会渲染成
  字面乱码；且 xindy 无法对中文条目正确排序。HTML 版的 genindex 与搜索不受影响。
