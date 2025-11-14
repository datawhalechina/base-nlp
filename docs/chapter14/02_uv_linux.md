# 第二节 云服务器模型部署实战

在上一节中，我们成功地使用 FastAPI 在本地构建了一个功能完备的 NER 模型 API 服务。不过，这种在个人电脑上运行的开发服务，其主要目的是用于功能验证和调试。要让模型真正发挥价值，我们需要将其部署到云服务器上，使其能够稳定、可靠地对外提供服务。

本节将以**阿里云 ECS（Elastic Compute Service）** 为例，一步步指导你完成从服务器准备、环境配置到应用部署和持久化运行的全过程。在这个过程中，我们将引入一个现代化的高性能 Python 打包工具 **[uv](https://github.com/astral-sh/uv)**，来替代传统的 `pip` 和 `venv`，体验极速的依赖安装和环境管理。

## 一、云服务器准备

在部署开始之前，我们需要有一个拥有公网 IP 的云服务器。市面上有许多云服务商可供选择，如阿里云、华为云、Amazon Web Services (AWS)等，它们提供的 **ECS** (弹性计算服务) 或 **VPS** (虚拟专用服务器) 都可以满足需求。

对于新用户或在校学生，各大云厂商通常会提供可观的**新人优惠或学生优惠**，可以以极低的成本租用到一台服务器。作为学习和实验用途，可以选择**按量计费**模式，这样只在服务器运行时才会计费，能有效控制成本。在这种模式下，不使用服务器时务必记得及时关机，避免产生不必要的费用。

在创建服务器时，需要设置登录凭证。通常有两种方式：

1.  **密码登录**: 这是最直接的方式，设置一个复杂的密码即可。简单方便，适合初学者和快速测试。
2.  **SSH 密钥对登录**: 这是更安全方式。它会生成一对密钥，**公钥**存放在服务器上，**私钥**（一个 `.pem` 或 `.key` 文件）需要下载并妥善保管在自己的电脑上。连接时，客户端会使用私钥来与服务器上的公钥进行匹配验证，全程无需在网络上传输密码，可以有效防止针对密码的暴力破解攻击。

> 笔者本次使用的是一台闲置的阿里云服务器，配置为 2核 CPU 和 2GB 内存，操作系统为 Ubuntu 22.04 LTS。这个配置对于我们本次的 NER 模型部署来说已经足够了。

无论选择哪家服务商，请务必在服务器的**网络防火墙**设置中，添加入站规则，放行以下两个端口。这个防火墙可能被称为**安全组**（常见于阿里云、AWS 等大型云厂商的 ECS 服务）或**云防火墙**（常见于 Vultr、DigitalOcean 等 VPS 服务商）。如果你的服务商没有提供云端防火墙，那么访问控制就落在了系统自带的防火墙上（如 `ufw` 或 `iptables`）。不过，在新创建的服务器上，系统防火墙通常默认是关闭的，所以服务可能无需额外配置即可访问。

-   **22**: 用于 SSH 远程登录。（大部分服务商的默认安全组会放行此端口，以便用户首次连接）
-   **8000**: 我们 FastAPI 应用将要监听的端口。

<p align="center">
  <img src="./images/14_2_1.png" width="80%" alt="阿里云 ECS 实例管理页面" />
  <br />
  <em>图 14-4 阿里云 ECS 实例管理页面</em>
</p>

准备好服务器后，会获得一个公网 IP 地址，后续我们将通过这个 IP 进行远程操作。登录云服务商的控制台，进入 ECS 实例管理页面如图 14-4，先找到并复制服务器的**公网 IP 地址**，后续连接服务器会用到。

接下来，以阿里云为例，演示如何配置安全组并放行 8000 端口。

1.  在实例管理页面的左侧导航栏中，找到并点击“网络与安全”下的“安全组”。

<p align="center">
  <img src="./images/14_2_2.png" width="80%" alt="安全组列表页面" />
  <br />
  <em>图 14-5 安全组列表页面</em>
</p>

2.  在安全组列表中，找到实例所绑定的安全组，点击右侧的“管理规则”。

3.  进入规则管理页面后，在“入方向”的选项下点击“增加规则”按钮。

<p align="center">
  <img src="./images/14_2_3.png" width="80%" alt="新建安全组入方向规则" />
  <br />
  <em>图 14-6 新建安全组入方向规则</em>
</p>

4.  如图 14-6 所示，在弹出的“新建安全组规则”窗口中，进行如下配置：
    -   **协议类型**: 选择“自定义 TCP”。
    -   **端口范围**: 填写 `8000`。
    -   **授权对象**: 为了方便测试，可以设置为 `0.0.0.0/0`，这表示允许任何 IP 地址访问。出于安全考虑，更推荐的做法是仅授权给你自己的 IP 地址。
    -   点击“提交”保存规则。

完成以上步骤后，我们就成功地为 FastAPI 应用放行了 8000 端口。

## 二、连接云服务器

获取到服务器的公网 IP 和登录凭证后，就可以远程连接并开始操作了。连接服务器的方式多种多样，可以根据自己的习惯和操作系统进行选择：

1.  **云服务商控制台**: 大部分云服务商都提供了网页版的远程连接工具（如 VNC），无需任何本地软件即可连接。这种方式方便快捷，适合临时性的简单操作。
2.  **标准 SSH 客户端**: 这是最经典、通用的方式。
    -   在 Windows 上，Windows 10 及以上已内置 OpenSSH 客户端，可直接在命令提示符或 PowerShell 中使用。此外，也可以选择 `PuTTY`、`Xshell` 等图形化工具。
    -   在 macOS 和 Linux 上，可以直接使用系统自带的终端（Terminal），通过 `ssh root@<公网IP>` 命令进行连接。
3.  **IDE 集成插件**: 许多现代化的代码编辑器和 IDE（如 VS Code, PyCharm）都提供了远程开发插件，可以直接连接到服务器，并在本地环境中无缝地进行远程文件编辑和代码调试。

对于需要频繁管理服务器的开发者来说，一款功能强大的集成式终端工具能极大地提升效率。本次我们选用 **FinalShell** 作为演示工具，这是一款免费且功能丰富的 SSH 客户端，集成了服务器监控、文件管理、命令历史等多种实用功能。

接下来，以 FinalShell 为例，介绍从安装到连接服务器的全过程。

### 2.1 下载与安装 FinalShell

首先，访问 FinalShell 的官方网站（[http://www.hostbuf.com/](https://www.hostbuf.com/t/988.html))，根据你的操作系统（Windows, macOS, Linux）下载对应的安装包。

<p align="center">
  <img src="./images/14_2_4.png" width="80%" alt="FinalShell 官网下载页面" />
  <br />
  <em>图 14-7 FinalShell 官网下载页面</em>
</p>

下载后，按照常规软件的安装步骤进行安装即可。

### 2.2 在 FinalShell 中创建连接

1.  打开 FinalShell，点击左上角的“文件夹”图标，打开“连接管理器”。

<p align="center">
  <img src="./images/14_2_5.png" width="80%" alt="FinalShell 连接管理器" />
  <br />
  <em>图 14-8 FinalShell 连接管理器</em>
</p>

2.  在弹出的窗口中，如图 14-8 点击第一个带加号的图标（“新建连接”），选择“SSH连接(Linux)”。

<p align="center">
  <img src="./images/14_2_6.png" width="80%" alt="FinalShell 新建 SSH 连接" />
  <br />
  <em>图 14-9 FinalShell 新建 SSH 连接</em>
</p>

3.  如图 14-9 所示，在弹出的“新建 SSH 连接”窗口中，填写服务器的相关信息：
    -   **名称**: 给连接起一个有意义的名称，方便识别其用途，例如 “阿里云-NER模型”。
    -   **主机**: 填入服务器公网 IP 地址。
    -   **端口**: 保持默认的 22。
    -   **认证方法**: 选择“密码”。
    -   **用户名**: 除非创建时有专门设置，否则默认为 `root`。
    -   **密码**: 填写你设置的服务器登录密码。

4.  填写完毕后，点击“确定”。此时，新的服务器配置会出现在连接管理器中。

<p align="center">
  <img src="./images/14_2_7.png" width="80%" alt="FinalShell 快速连接" />
  <br />
  <em>图 14-10 FinalShell 快速连接</em>
</p>

5.  如图 14-10，双击刚刚创建的连接，FinalShell 就会开始尝试连接你的云服务器。首次连接时，可能会弹出一个接受主机密钥的提示，点击“接受并保存”即可。

<p align="center">
  <img src="./images/14_2_8.png" width="80%" alt="FinalShell 命令行界面" />
  <br />
  <em>图 14-11 FinalShell 命令行界面</em>
</p>

连接成功后，如图 14-11 能看到一个 Linux 命令行界面，同时窗口左侧还会动态显示服务器的 CPU、内存和网络使用情况。至此，我们已经成功登录到了云服务器，可以开始配置环境了。

## 三、uv 项目管理工具

在开始部署之前，先来认识一款能够显著提升效率的工具 `uv`。`uv` 是一个用 Rust 编写的极速 Python 包安装器和解析器，由 `ruff` 的作者开发，目标是作为 `pip`、`pip-tools`、`venv` 等工具的统一、高速替代品。它的速度非常快，尤其是在解析和安装有复杂依赖关系的包时，能节省大量时间。在本次部署中，我们会全程使用 `uv` 来管理虚拟环境和项目依赖。

### 3.1 `uv` 的项目配置 (`pyproject.toml`)

`uv` 遵循 PEP 621 标准，支持通过项目根目录下的 `pyproject.toml` 文件进行配置。这使得项目依赖和工具配置能够集中管理，提高了项目的可移植性和可复现性。在一个包含 `pyproject.toml` 的项目中运行 `uv` 命令（如 `uv pip install`）时，`uv` 会自动读取其中的配置。

我们可以在 `pyproject.toml` 文件中定义一个 `[tool.uv]` 表，来存放所有 `uv` 相关的配置。以下是一些常用的配置选项：

-   **`native-tls`**: 一个布尔值，用于控制是使用 `native-tls`（系统原生 TLS 实现）还是 `rustls`（跨平台 TLS 实现）。默认值为 `false`（使用 `rustls`）。在某些网络环境下，切换到 `native-tls` 可能有助于解决 SSL/TLS 相关的连接问题。
-   **`index-url`**: 指定默认的 PyPI 索引 URL，相当于 `pip` 的 `--index-url`。
-   **`extra-index-url`**: 指定额外的 PyPI 索引 URL，相当于 `pip` 的 `--extra-index-url`。这对于同时使用公共 PyPI 和私有镜像源的场景非常有用。
-   **`no-index`**: 一个布尔值，设为 `true` 时，`uv` 将不会使用任何包索引，仅依赖于 `find-links` 指定的路径或已有的 `uv.lock` 文件。
-   **`find-links`**: 一个 URL 或本地路径的列表，用于指定查找包的备用位置。

下面就以本次部署的 NER 项目为例，为它编写一个完整的 `pyproject.toml` 文件：

```toml
# ner_deployment/pyproject.toml

[project]
name = "ner-deployment-service"
version = "1.0.0"
description = "NER model deployment with FastAPI and uv."
dependencies = [
    "fastapi",
    "pydantic",
    "uvicorn",
    "gunicorn",
    "torch==2.2.1",
]

[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
```

在这个配置文件中，我们不仅在 `[project]` 部分定义了项目的基本信息和依赖，更关键的是，在 `[tool.uv]` 部分，我们使用了 `[[tool.uv.index]]` 表来配置包索引源。这种写法是当前 `uv` 推荐的方式，它取代了已被弃用的 `index-url` 和 `extra-index-url` 字段：
1.  **定义多个源**：我们定义了两个名为 `aliyun` 和 `pytorch-cpu` 的索引源。
2.  **优先级**：`uv` 会按照它们在文件中出现的顺序进行查找。首先会去**阿里云镜像**寻找所有包，如果在该源找不到某个包（或者特定版本），它会自动去下一个源，即 **PyTorch 官方 CPU 索引**中寻找。

这种配置方式比旧的写法更加清晰，且扩展性更好，完美解决了既要享受国内镜像的速度，又要能下载到特殊官方源的包的问题。

通过将这样的配置文件与项目代码一同托管，可以确保团队中的每个成员以及 CI/CD 流水线都使用完全相同的依赖源和配置，从而避免因环境不一致导致的问题。

## 四、配置服务器环境

成功连接到服务器后，我们就拥有了一个全新的、待配置的 Linux 环境。在部署应用之前，需要完成一系列准备工作，包括更新系统、安装必要的工具，并为项目创建一个隔离的运行环境。

### 4.1 系统更新

登录服务器后，首先更新软件包列表和已安装的包，这是一个好习惯：

```bash
sudo apt update && sudo apt upgrade -y
```

### 4.2 安装 Python 与 uv

`uv` 本身是一个独立的二进制文件，它支持多种安装方式。在本教程中，我们选择通过 `pip` 来安装和管理它，需要服务器上先有一个 Python 环境。

1.  **检查 Python 版本**

    首先，检查服务器上是否已安装 Python，以及其版本。

    ```bash
    python3 --version
    ```
    
    笔者使用的服务器自带了 Python 3.10.12，可以满足需求。如果读者的服务器没有安装 Python，或者版本过低，可以使用 `apt` 来安装。例如，安装 Python 3.10：

    ```bash
    sudo apt install -y python3.10
    ```

2.  **安装 uv**

    有了 Python 环境后，就可以使用 `pip` 来安装 `uv` 了：

    ```bash
    pip install uv
    ```

    安装完成后，可以通过以下命令验证安装是否成功：

    ```bash
    uv --version
    ```

### 4.3 准备项目文件与环境

现在，我们来为部署项目创建一个独立的虚拟环境。

1.  **上传项目文件**

    直接将本地的 `ner_deployment` 文件夹上传到服务器的用户主目录（通常是 `/root`）。

    利用 FinalShell 的图形化文件管理功能，操作非常简单。在 FinalShell 的文件浏览器中，确保服务器的路径是在用户主目录下，然后直接将本地的 `code/C14/ner_deployment` 目录拖拽到下方的服务器文件列表中即可。

    上传完成后，进入项目目录：
    ```bash
    cd ner_deployment
    ```
    
    <p align="center">
       <img src="./images/14_2_9.png" width="80%" alt="FinalShell 文件上传" />
       <br />
       <em>图 14-12 FinalShell 文件上传</em>
    </p>

2.  **创建并激活虚拟环境**

    在项目根目录下，运行以下命令。`uv` 会自动创建一个名为 `.venv` 的虚拟环境。

    ```bash
    uv venv
    ```

    激活虚拟环境:

    ```bash
    source .venv/bin/activate
    ```
    
    激活后，终端提示符前面会出现 `(ner_deployment) `，表示当前正处于这个虚拟环境中。后续所有的 Python 和 `uv` 命令都将作用于此环境内，与系统环境隔离。

## 五、部署 NER 应用

项目文件和虚拟环境都准备就绪后，我们就可以开始安装依赖并运行服务了。

### 5.1 安装依赖

由于我们已经在 `ner_deployment` 目录中提供了 `pyproject.toml` 文件，并声明了所有项目依赖，所以可以直接使用 `uv` 从该文件进行安装。

我们将项目以“可编辑模式”进行安装。这不仅会自动安装 `pyproject.toml` 中定义的所有依赖，还会将当前项目本身也视为一个已安装的包。

```bash
uv pip install -e .
```

> **处理网络超时**
> 在网络状况不佳的服务器上，`uv` 的默认 30 秒超时可能不足以完成较大依赖包的下载，并导致安装失败，出现类似 `Failed to download distribution due to network timeout` 的错误。
> 此时，可以按照错误提示，通过设置 `UV_HTTP_TIMEOUT` 环境变量来延长超时时间。例如，将超时设置为 300 秒（5分钟）：
> ```bash
> UV_HTTP_TIMEOUT=300 uv pip install -e .
> ```

`uv` 会自动读取 `pyproject.toml` 文件，解析依赖，并从我们配置好的源（包括阿里云镜像和 PyTorch CPU 源）高速下载并安装。

### 5.2 启动生产服务

在开发时，使用的是 `uvicorn main:app --reload` 来启动服务。这个命令默认只允许本地访问（监听 `127.0.0.1`）。如果希望在云服务器上临时用它进行测试，并从外部访问，则必须指定 `--host 0.0.0.0`。但在生产环境中，这种方式不够健壮，我们需要一个更专业的方案来管理应用进程，并确保服务在后台稳定运行。

`Gunicorn` 是一个成熟的 Python WSGI HTTP 服务器，我们用它来管理 `Uvicorn`工作进程，这是一个常见的生产部署模式。

我们可以测试一下能否正常启动：

```bash
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

-   `-w 3`: 启动 3 个工作进程（worker）。Gunicorn 官方文档推荐的通用设置是 `2 * CPU核心数 + 1`。但对于我们所使用的 `UvicornWorker` 这种异步工作进程，由于其高效的并发处理能力，通常设置为 `CPU核心数 + 1` 就足够了。我们的服务器是 2 核 CPU，所以设置为 `3`。
-   `-k uvicorn.workers.UvicornWorker`: 指定 Gunicorn 使用 Uvicorn 的工作进程类，以便支持 `asyncio`。
-   `--bind 0.0.0.0:8000`: 绑定到 `0.0.0.0`，意味着服务器将监听所有可用的 IP 地址上的 8000 端口，从而允许外网访问。

<p align="center">
  <img src="./images/14_2_10.png" width="80%" alt="Gunicorn 启动日志" />
  <br />
  <em>图 14-13 Gunicorn 启动日志</em>
</p>

如果看到如图 14-13 服务成功启动的日志，说明应用本身没有问题。按 `Ctrl+C` 停止它。

### 5.3 使用 Systemd 持久化服务

直接在终端运行 `gunicorn` 命令，一旦关闭 SSH 连接，服务就会中断。为了让我们的 API 服务能在后台长期运行，并且在服务器重启后能自动启动，需要使用 `systemd`——Linux 系统标准的服务管理器。

1.  **创建 systemd 服务文件**:

    ```bash
    sudo nano /etc/systemd/system/ner_api.service
    ```

2.  **编写服务配置**:
    在打开的 `nano` 编辑器中，粘贴以下内容。**注意：** 你需要将 `User` 和 `WorkingDirectory`、`ExecStart` 中的路径替换为你自己的实际用户名和项目路径。

    ```ini
    [Unit]
    Description=NER API Service
    After=network.target

    [Service]
    User=root
    Group=root
    WorkingDirectory=/root/ner_deployment
    ExecStart=/root/ner_deployment/.venv/bin/gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
    Restart=on-failure
    RestartSec=5s

    [Install]
    WantedBy=multi-user.target
    ```

    -   `[Unit]`: 定义了服务的元数据和依赖关系。`After=network.target` 表示服务应在网络准备好之后启动。
    -   `[Service]`: 定义了服务的核心行为。
        -   `User`/`Group`: 运行服务的用户和组。
        -   `WorkingDirectory`: 启动服务前切换到的工作目录。
        -   `ExecStart`: 启动服务的完整命令。**务必使用虚拟环境中的 `gunicorn` 绝对路径**。
        -   `Restart`: 设置服务在失败时自动重启。
    -   `[Install]`: 定义了服务的安装信息。`WantedBy=multi-user.target` 表示服务应在系统以多用户模式启动时启用。

    编辑完成后，按 `Ctrl+X`，然后按 `Y` 保存，最后按 `Enter` 确认。

3.  **管理服务**:
    现在，使用 `systemctl` 命令来启动并管理新服务。由于我们在服务文件中已经指定了虚拟环境的绝对路径，并且 `sudo` 会以 `root` 权限执行，所以**无需刻意进入或退出虚拟环境**，在任何目录下执行以下命令效果都是一样的。

    -   **重新加载 systemd 配置**：让 systemd 读取新的服务文件。
        ```bash
        sudo systemctl daemon-reload
        ```
    -   **启动服务**:
        ```bash
        sudo systemctl start ner_api
        ```
    -   **设置开机自启**:
        ```bash
        sudo systemctl enable ner_api
        ```
    -   **查看服务状态**:
        ```bash
        sudo systemctl status ner_api
        ```
        
        <p align="center">
          <img src="./images/14_2_11.png" width="80%" alt="systemctl status 显示服务正在运行" />
          <br />
          <em>图 14-14 systemctl status 显示服务正在运行</em>
        </p>

        如果看到如图 14-14 所示 `active (running)` 的绿色字样，说明服务已成功部署并正在后台运行！

要查看服务的实时日志，可以使用 `journalctl` 命令：

```bash
sudo journalctl -u ner_api -f
```

### 5.4 测试云端服务

现在，服务已经部署在云端。在**自己的电脑**上打开一个新的终端，使用 `curl` 命令来测试它，将 `<服务器公网IP>` 替换为你的实际 IP。

```bash
curl -X POST "http://<服务器公网IP>:8000/predict/ner" -H "Content-Type: application/json" -d "{\"text\":\"患者自述发热、咳嗽，伴有轻微头痛。\"}"
```

> **注意**：在 Windows PowerShell 中，`curl` 是 `Invoke-WebRequest` 命令的别名，其参数格式与标准 `curl` 不同，直接运行以上命令会报错。推荐在 `cmd` 或 `bash` 环境（如 Git Bash）中执行此命令。

<p align="center">
  <img src="./images/12_2_12.png" width="80%" alt="curl 测试云端服务" />
  <br />
  <em>图 14-15 curl 测试云端服务</em>
</p>

如果一切顺利，会收到如图 14-12 和本地测试时一致的 JSON 响应，里面包含了模型识别出的实体。同时，也可以在浏览器中访问 `http://<服务器公-网IP>:8000/docs`，来查看并使用 FastAPI 自动生成的交互式 API 文档。