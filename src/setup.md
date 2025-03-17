# Setup

本书使用mdbook构建，托管于github.io，以WSL环境为例，记录一下构建过程。

## 安装

### 安装WSL

略

---

打开WSL

### 安装Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
验证安装是否成功：
```bash
rustc --version
```

### 安装mdbook

```bash
cargo install mdbook
```

## 构建并运行

### 创建mdbook项目

```bash
mdbook init paper
```
- 是否需要`.gitignore`文件：`y`
- 输入项目名称：`paper_reading`（后续可在`book.toml`中更改）

### 构建mdbook
在`/paper`目录下执行：
- 构建项目：
```bash
mdbook build
```
- 或是在浏览器中实时预览：
```bash
mdbook serve
```

## 部署
1. 新建Github仓库，将项目上传至仓库。
2. 在顶栏目录中找到`Actions`，搜索`mdbook`，点击`Configure`，自动生成`.yml`文件。点击`Commit Changes`提交。
3. 在顶栏目录中找到`Settings`，在侧边栏中找到`Pages`，在`Build and deployment`下找到`Source`，选择`Github Actions`。
4. 第二步会在`/paper`下创建`./.github/workflows/mdbook.yml`文件，在本地pull更改。
5. 之后本地修改内容后push到仓库，Github Actions会自动构建并部署。访问`https://<username>.github.io/<reponame>/`即可查看。

## 参考
- [安装Rust](https://www.rust-lang.org/tools/install)
- [mdbook文档](https://rust-lang.github.io/mdBook/)