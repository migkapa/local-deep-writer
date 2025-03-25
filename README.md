# Local Deep Writer

Local Deep Writer is a powerful article generation tool that creates comprehensive, SEO-optimized content from web research. Built on top of local language models, it produces well-structured articles with proper HTML formatting, detailed table of contents, and organized sections.

This project is inspired by and modified from [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher), but specifically focused on generating high-quality articles rather than just research summaries.

## Features

- Generates in-depth articles (1500-2000+ words)
- Creates structured content with table of contents and anchor links
- Organizes content into 5-7 well-defined sections
- Includes proper HTML formatting for web publishing
- Optimizes content with SEO keywords
- Uses local LLMs for content generation
- Leverages web research to ensure accurate, up-to-date content

## 🚀 Quickstart

1. Clone the repository:
```shell
git clone https://github.com/migkapa/local-deep-writer.git
cd local-deep-writer
```

2. Set up your environment:
```shell
cp .env.example .env
```
Edit the `.env` file to configure your:
- Search API preference (duckduckgo, tavily, perplexity, or searxng)
- LLM provider (ollama or lmstudio)
- Model settings and API endpoints

### Option 1: Using Ollama

1. Download [Ollama](https://ollama.com/download) for your platform

2. Pull a local LLM (we recommend DeepSeek for best results):
```shell
ollama pull deepseek-r1:8b
```

3. Configure Ollama in your `.env`:
```shell
LLM_PROVIDER=ollama
OLLAMA_BASE_URL="http://localhost:11434"
LOCAL_LLM=deepseek-r1:8b
```

### Option 2: Using LMStudio

1. Download [LMStudio](https://lmstudio.ai/)
2. Start your selected model in LMStudio
3. Configure LMStudio in your `.env`:
```shell
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL="http://localhost:1234"
LOCAL_LLM=deepseek-r1-distill-llama-8b
```

## Usage

1. Install dependencies:
```shell
pip install -e .
```

2. Start the application:
```shell
langgraph dev
```

3. Visit the LangGraph Studio UI:
[http://127.0.0.1:2024](http://127.0.0.1:2024)

## Example Output

The tool generates articles with:
- HTML-formatted content
- Table of contents with anchor links
- 5-7 well-structured sections
- SEO-optimized headers and content
- 1500-2000+ words of detailed content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is built upon and inspired by [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher), modified to focus specifically on article generation with enhanced structure and SEO optimization.


## Outputs

The output of the graph is a markdown file containing the research summary, with citations to the sources used. All sources gathered during research are saved to the graph state. You can visualize them in the graph state, which is visible in LangGraph Studio:

![Screenshot 2024-12-05 at 4 08 59 PM](https://github.com/user-attachments/assets/e8ac1c0b-9acb-4a75-8c15-4e677e92f6cb)

The final summary is saved to the graph state as well:

![Screenshot 2024-12-05 at 4 10 11 PM](https://github.com/user-attachments/assets/f6d997d5-9de5-495f-8556-7d3891f6bc96)

## Deployment Options

There are [various ways](https://langchain-ai.github.io/langgraph/concepts/#deployment-options) to deploy this graph. See [Module 6](https://github.com/langchain-ai/langchain-academy/tree/main/module-6) of LangChain Academy for a detailed walkthrough of deployment options with LangGraph.

## TypeScript Implementation

A TypeScript port of this project (without Perplexity search) is available at:
https://github.com/PacoVK/ollama-deep-researcher-ts

## Running as a Docker container

The included `Dockerfile` only runs LangChain Studio with local-deep-researcher as a service, but does not include Ollama as a dependant service. You must run Ollama separately and configure the `OLLAMA_BASE_URL` environment variable. Optionally you can also specify the Ollama model to use by providing the `LOCAL_LLM` environment variable.

Clone the repo and build an image:
```
$ docker build -t local-deep-researcher .
```

Run the container:
```
$ docker run --rm -it -p 2024:2024 \
  -e SEARCH_API="tavily" \ 
  -e TAVILY_API_KEY="tvly-***YOUR_KEY_HERE***" \
  -e LLM_PROVIDER=ollama
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434/" \
  -e LOCAL_LLM="llama3.2" \  
  local-deep-researcher
```

NOTE: You will see log message:
```
2025-02-10T13:45:04.784915Z [info     ] 🎨 Opening Studio in your browser... [browser_opener] api_variant=local_dev message=🎨 Opening Studio in your browser...
URL: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024
```
...but the browser will not launch from the container.

Instead, visit this link with the correct baseUrl IP address: [`https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024`](https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024)
