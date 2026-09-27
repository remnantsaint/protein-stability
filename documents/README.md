# 蛋白质稳定性预测：FastAPI 本地版

项目目录：`D:\Projects\WEB`。2026-09-18 从 Flask 重构为 FastAPI，保留已有页面、模型权重、网络结构和突变校验规则。

访问地址：<http://127.0.0.1:8088/>。接口文档：<http://127.0.0.1:8088/docs>。健康检查：<http://127.0.0.1:8088/healthz>。

## 你先看哪些文件

```text
app.py                  FastAPI：输入模型、路由、生命周期、异常处理
inference.py            从原 app.py 移出的模型加载、特征提取、预测流程
mutation_utils.py       原有序列与突变校验逻辑，未改动
model.py / model.pt     原有网络定义与权重，未改动
templates/index.html    原有网页，调整了静态图片的模板参数
static/figure.jpg       网页中的模型结构图
tests/test_app.py       FastAPI 接口、异常、生命周期测试
requirements.txt        本次使用的主要依赖版本
.venv/                  本项目独立 Python 环境
.local/                 日志、验证结果、原版备份等，无需作为业务代码阅读
```

没有新增一组 cmd 或 ps1 启动脚本。先读 app.py 中的 PredictionRequest 和 predict，再读 lifespan，最后看 inference.py 的 ModelService。

## 本地运行

在 PowerShell 中执行：

```powershell
cd D:\Projects\WEB
.\.venv\Scripts\python.exe app.py
```

看到 `Application startup complete` 后打开页面。前台运行时按 Ctrl+C 停止。模型加载可能需要几秒到更久，取决于磁盘和运行环境。

也可以显式使用 Uvicorn，两个命令任选一个，不要同时运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8088 --workers 1
```

本次已经在后台启动了 8088 端口的实例，直接访问即可；同端口不能再启动第二个实例。当前后台日志在 `.local/server.stderr.log` 和 `.local/server.stdout.log`。前台运行时日志直接显示在终端。

本机安装的是 CPU 版 PyTorch，真实预测在 CPU 执行；换到 CUDA 环境时，需要先安装与目标环境匹配的 GPU 版 PyTorch。代码通过 `torch.cuda.is_available()` 选择设备。

不要直接增加 worker 数：每个应用进程都会各自加载 ESM 和预测模型。没有启用自动重载，修改服务代码后需重启。

## 改了哪些

| 原来 | 现在 | 作用 |
|---|---|---|
| Flask / app.route / jsonify | FastAPI / app.get、app.post / response_model | 定义接口及返回字段，自动生成接口文档 |
| data.get 手动取值 | PredictionRequest + Pydantic 校验器 | 明确类型并校验序列、突变格式、位置与原始氨基酸 |
| 模型只在 __main__ 中加载 | lifespan 启动时加载，关闭时释放 | 用 Uvicorn 导入 app 也能正确初始化；加载失败则启动失败 |
| 模型保存在几个全局变量中 | app.state.service 保存一个 ModelService | 多个请求复用同一进程内的模型对象 |
| 路由和推理混在 app.py | app.py 负责接口，inference.py 负责推理 | 阅读时能把 Web 与模型流程分开 |
| model.pt 使用相对工作目录 | 根据源码文件定位权重与模板 | 从其他目录启动也能找到资源 |
| 异常字符串直接发给用户 | 日志记录内部异常，返回通用 500 错误 | 用户看到明确错误，内部细节留在日志 |
| 没有显式限制同时推理数 | 单进程一次处理一个推理，忙时返回 503 | 避免本地多个推理请求同时挤占内存；这不是后台任务队列 |
| Flask 模板 filename 参数 | Starlette 模板 path 参数 | 继续显示原有静态图片 |
| Flask 测试客户端 | FastAPI TestClient + 可替换模型工厂 | 接口测试不需要加载大型权重 |

模型的特征计算与推理顺序保持一致：原始序列和突变序列分别提取 ESM 特征，再输入已有 Predictor。没有重新训练或替换 model.pt。

## 请求流程

```text
启动 Uvicorn
    → lifespan 创建 ModelService，加载 ESM 与 model.pt
    → 开始接收请求

网页 POST /predict
    → PredictionRequest 校验并规范化输入
    → Depends(get_service) 取得已加载的模型对象
    → 检查是否已有请求正在推理
    → ModelService.predict 执行预测
    → PredictionResponse 校验返回字段
    → 返回 JSON，网页显示结果

停止服务
    → lifespan 执行 finally，释放模型引用
```

推理接口使用普通 `def`，由 FastAPI 放到线程池执行。加上 `async` 关键字本身不会让 PyTorch 推理变快。并发占用控制只在单进程内生效，所以本地使用一个 worker。

## 接口与兼容性

- `GET /` 与 `GET /ddg-predictor/`：原有网页。
- `POST /predict` 与 `POST /ddg-predictor/predict`：预测，返回字段仍为 ddg、length、mutation、mutation_count。
- `GET /healthz`：模型已加载状态及当前设备。
- `GET /docs`、`GET /openapi.json`：接口说明与机器可读定义。
- 新请求体：`{"sequence": "...", "mutations": "F88Y_L91A"}`。
- 旧请求体仍支持：`{"sequence": "...", "pos": 88, "wt": "F", "mt": "Y"}`。
- 同时提供新旧字段时，非空 mutations 优先，与原接口一致。
- 非法 JSON、数组请求体、错误字段类型、重复位置、越界、原始氨基酸不匹配、超过 10 个突变均返回 400。
- 推理失败返回 500；模型正忙返回 503 和 Retry-After，不会自动重复提交。

新版本对字段类型更严格，数组和数字不会再被随意转成字符串。错误格式仍保持 `{"error": "说明"}`，原网页能直接显示。

## 亲手试一次

网页中输入下面的序列，Mutation set 填 `F88Y_L91A`：

```text
MTEFKAGSAKKGATLFKTRCLQCHTVEKGGPHKVGPNLHGIFGRHSGQAEGYSYTDANIKKNVLWDENNMSEYLTNPKKYIPGTKMAFGGLKKEKDRNDLITYLKKACE
```

也可直接在 PowerShell 调用接口：

```powershell
$body = @{
    sequence = 'MTEFKAGSAKKGATLFKTRCLQCHTVEKGGPHKVGPNLHGIFGRHSGQAEGYSYTDANIKKNVLWDENNMSEYLTNPKKYIPGTKMAFGGLKKEKDRNDLITYLKKACE'
    mutations = 'F88Y_L91A'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8088/predict' -ContentType 'application/json' -Body $body
```

本机本次真实结果为 `ddg=-2.5934`、`length=109`、`mutation_count=2`。页面使用已有的外部 Vue/Element UI 等资源，需要能访问相应 CDN。

## 验证结果

- 13 项接口与生命周期测试通过（部分测试包含多个输入子案例），模型使用模拟对象。
- 另外加载真实 ESM 权重和 model.pt，执行 3 次 HTTP 预测，对比原 Flask 文件中的推理函数：单点 F88Y、双点 F88Y_L91A、旧字段单点输入。
- 单点输出 -0.5218，双点输出 -2.5934；三次与原代码在接口四位小数精度下一致。
- 本机这三次 HTTP 推理约 1.4～1.9 秒，仅代表此 109 残基样例和当前 CPU 环境，不是通用性能结论。
- 两个页面路径、静态图片、接口文档、OpenAPI、健康检查实际 HTTP 返回正常。
- pip check 通过。原始验证记录：`.local/verification.json`。

自行运行接口测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

原 Flask 文件备份在 `.local/before-fastapi/`。本次范围是 FastAPI 重构与本机运行，Nginx 反向代理尚未配置。

## 依赖与模型缓存

目前环境已准备好，不用重新安装。若将来重建 CPU 环境，可先安装 CPU 版 PyTorch，再安装 requirements.txt：

```powershell
D:\Anaconda\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机复用了已有的用户目录 Torch/ESM 缓存，没有重新下载约 2.6 GB 的 ESM 主权重。源码优先尊重 TORCH_HOME；未指定且没有现成模型缓存时，会把下载位置设置到项目 `.local/torch/hub`。

学习参考：[FastAPI 生命周期](https://fastapi.tiangolo.com/advanced/events/)、[模板](https://fastapi.tiangolo.com/advanced/templates/)、[生命周期测试](https://fastapi.tiangolo.com/advanced/testing-events/)、[PyTorch 安装](https://pytorch.org/get-started/locally/)。
