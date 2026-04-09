# GitHub 组员协作使用指南

这份文档是写给**几乎没有用过 GitHub** 的组员看的。  
如果你以前没接触过 `git`、`GitHub`、`branch`、`pull request`，直接按下面步骤做就可以。

补充约定：当前项目中的数据切分与训练脚本默认随机种子统一使用 `4210`，提交实验结果时请保持一致。

## 1. 先理解几个最基本的概念

### 什么是 Git

`Git` 是一个版本管理工具。  
你可以把它理解成“代码的存档系统”，它可以记录：

- 谁改了什么
- 什么时候改的
- 改坏了以后怎么找回
- 多个人怎么一起改同一个项目

### 什么是 GitHub

`GitHub` 是放 `Git` 仓库的网站。  
你可以把它理解成：

- 一个共享代码的平台
- 一个组员一起协作的地方
- 一个用来提交、查看、合并代码的网站

### 什么是仓库

仓库就是这个项目的文件夹，只不过它被 `Git` 管理了。  
你们这个课程项目就是一个仓库。

### 什么是分支

分支可以理解成“单独开一条线开发”。  
每个人最好在自己的分支上改，改完以后再合并到主分支 `main`。

这样做的好处是：

- 不会直接把主分支改乱
- 大家互相不会轻易覆盖代码
- 哪个功能是谁写的很清楚

## 2. 组员协作的整体流程

你只要记住下面这个最核心流程：

1. 接受仓库邀请
2. 把仓库下载到自己电脑上
3. 新建自己的分支
4. 在自己的分支上修改文件
5. 提交改动
6. 推送到 GitHub
7. 发起 `Pull Request`
8. 由组长或其他组员合并到 `main`

## 3. 第一次使用前要准备什么

你需要：

- 一个 GitHub 账号
- 电脑上装好 `git`
- 能打开终端

### 检查电脑有没有安装 git

在终端输入：

```bash
git --version
```

如果能看到版本号，比如：

```bash
git version 2.xx.x
```

就说明已经装好了。

如果没有装好，可以去官网安装：

- [Git 官网](https://git-scm.com/)

## 4. 组长需要先做什么

组长需要在 GitHub 上把组员加进仓库。

大致步骤：

1. 打开仓库页面
2. 点击 `Settings`
3. 找到 `Collaborators` 或 `Manage access`
4. 输入组员的 GitHub 用户名
5. 发出邀请

组员接受邀请以后，才可以一起协作。

## 5. 组员第一次如何开始

### 第一步：接受 GitHub 邀请

组长邀请你后，GitHub 会有通知。  
点接受即可。

### 第二步：把仓库下载到自己电脑

在终端里进入你想放项目的目录，然后输入：

```bash
git clone <仓库地址>
```

例如：

```bash
git clone https://github.com/xxx/xxx.git
```

下载完成后进入项目目录：

```bash
cd DDA4210
```

注意：

- `clone` 只需要第一次做一次
- 以后就不用重复 `clone`

## 6. 第一次进入项目后该做什么

先看看当前状态：

```bash
git status
```

再拉一下最新主分支：

```bash
git pull origin main
```

然后创建你自己的分支：

```bash
git checkout -b feat-your-task
```

比如你负责双流模型，可以叫：

```bash
git checkout -b feat-dual-stream
```

比如你负责文档，可以叫：

```bash
git checkout -b docs-report-update
```

建议分支名简单清楚：

- `feat-xxx`：新功能
- `fix-xxx`：修 bug
- `docs-xxx`：文档修改

## 7. 日常开发怎么做

以后每次开发，基本都是这个流程。

### 第一步：先切到自己的分支

```bash
git branch
```

这个命令会显示你现在在哪个分支。  
带 `*` 的就是当前分支。

如果不在自己的分支上，就切过去：

```bash
git checkout feat-dual-stream
```

### 第二步：开始修改文件

你可以直接在 Cursor、VS Code 或其他编辑器里改代码。

### 第三步：查看哪些文件改了

```bash
git status
```

如果看到：

```bash
modified: 某个文件
```

说明这个文件改过了，但还没准备提交。

### 第四步：把改动加入暂存区

如果只想提交某一个文件：

```bash
git add "文件路径"
```

比如：

```bash
git add "README.md"
```

如果想把当前所有改动都加入：

```bash
git add .
```

### 第五步：提交改动

```bash
git commit -m "写清楚这次改了什么"
```

例如：

```bash
git commit -m "add HybridForensics evaluation notes"
git commit -m "implement first dual-stream model"
git commit -m "update project README"
```

注意：

- `commit` 前要先 `git add`
- 不然会出现“没有可提交内容”

### 第六步：推送到 GitHub

第一次推送你的新分支：

```bash
git push -u origin feat-dual-stream
```

以后如果还是这个分支，通常直接：

```bash
git push
```

## 8. 怎么把自己的改动交给组长

当你已经把分支推到 GitHub 后：

1. 打开 GitHub 仓库页面
2. GitHub 往往会提示你刚刚推了一个新分支
3. 点击 `Compare & pull request`
4. 填写标题和说明
5. 点击创建 `Pull Request`

### 什么是 Pull Request

可以把它理解成：

> “我这部分写好了，请大家检查一下，然后合并到主分支。”

组长或其他组员 review 后，就可以把你的分支合并到 `main`。

## 9. 每次开始新任务前要做什么

开始新任务前，先同步主分支最新内容。

### 推荐做法

先切回主分支：

```bash
git checkout main
```

拉取远程最新代码：

```bash
git pull origin main
```

再新建一个新的任务分支：

```bash
git checkout -b feat-new-task
```

不要长期在一个分支上把所有事情都混着做。  
最好一个任务一个分支。

## 10. 推荐的组内协作规则

为了减少混乱，建议所有人都遵守下面这些规则：

1. 不直接在 `main` 上开发
2. 每个人都在自己的分支上改
3. 一个任务一个分支
4. 提交信息写清楚
5. 推送后通过 `Pull Request` 合并
6. 合并前先同步最新 `main`
7. 不要把大数据、模型权重和输出结果直接传到 GitHub

## 11. 常见错误和解决方法

### 情况 1：改了文件但不能 commit

如果你看到类似：

```bash
no changes added to commit
```

意思是：

- 你虽然改了文件
- 但还没有 `git add`

解决方法：

```bash
git add .
git commit -m "your message"
```

### 情况 2：push 被拒绝

如果你看到远程拒绝，通常说明：

- 别人已经先推了更新
- 你的本地不是最新版本

先执行：

```bash
git pull origin main
```

如果你当前在自己的分支上，也可能需要：

```bash
git pull
```

处理完冲突后再 `git push`。

### 情况 3：发生冲突

冲突的意思是：

- 你改了某一段
- 别人也改了同一段
- Git 不知道该保留谁的版本

这时你需要：

1. 打开冲突文件
2. 手动决定保留哪部分内容
3. 删除冲突标记
4. 再执行：

```bash
git add .
git commit -m "resolve merge conflict"
```

如果你不会处理，先不要乱删，找组长一起看。

### 情况 4：不小心在 main 上改了

如果你还没提交，最简单的做法是：

```bash
git checkout -b feat-new-task
```

这会把当前改动一起带到新分支上。  
然后你就在这个新分支继续工作。

## 12. 最常用命令一览

```bash
git clone <仓库地址>
git status
git pull origin main
git checkout -b feat-your-task
git branch
git add .
git commit -m "your message"
git push -u origin feat-your-task
git push
```

## 13. 给第一次使用 GitHub 的同学的最简版本

如果你只想先记住最少内容，就记这 6 步：

1. `git clone <仓库地址>`
2. `cd DDA4210`
3. `git checkout -b feat-your-task`
4. 改文件
5. `git add . && git commit -m "your message"`
6. `git push -u origin feat-your-task`

然后去 GitHub 页面发 `Pull Request`。

## 14. 术语对照

- `clone`：把远程仓库下载到自己电脑
- `branch`：分支
- `main`：主分支
- `add`：把改动加入暂存区
- `commit`：把一次改动正式记录下来
- `push`：把本地提交上传到 GitHub
- `pull`：把 GitHub 上的新内容拉到本地
- `Pull Request`：申请把你的分支合并到主分支

## 15. 最后建议

刚开始不用一次把所有命令都记住。  
你只要先会下面这一套就够了：

```bash
git status
git add .
git commit -m "your message"
git push
```

如果不确定当前该做什么，先执行：

```bash
git status
```

这个命令通常最有用。
