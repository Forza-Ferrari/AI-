环境依赖：

```
conda create -n ai_env python=3.10
conda activate ai_env
pip install -r requirements.txt
```

数据准备（需要本地有 ```data/contexts.json```）：

```
python scripts/build_faiss.py
```

启动 CLI：

```
python main.py
```

启动 GUI：

```
streamlit run app.py
```

使用说明：

使用前需要在自行在根目录建立一个 ```.env``` 文件，内容如下：

```
DEEPSEEK_API_KEY=***
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/
```

分别存放自己使用的 APIkey 和 url。

输入医疗问题，例如“经常头晕，感到乏力怎么办“，”糖尿病在饮食方面的注意事项有哪些“，并等待回答即可。

注意事项：

请确保本机联网，DeepSeek API Key 正确设置。

如果遇到 FAISS 错误，请先运行 `build_faiss.py` 构建索引。

可能需要使用 VPN。
