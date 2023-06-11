<p align="center">
<img width="500px" alt="Aurora IA" src="https://i.redd.it/zkndu5kilmc31.png">
</p>

# Aurora IA 

Este repositorio utiliza a formatação, scripts e aplicação originais do projeto <a href="https://github.com/project-baize/baize-chatbot">Baize-Chatbot</a>, porém modificado com objetivo de gerar conteúdo enviesado para uma perspectiva específica. Além disso desenvolvemos um script de inferencia customizado para utilizar modelos ggml. Neste projeto nao utilizamos nenhum dos dados gerados pelo projeto original do Baize.
O script de ingestão dos arquivos foi completamente copiado do repositório <a href="https://github.com/PromtEngineer/localGPT">localGPT</a>.

Modelo lora <a href="https://huggingface.co/chenuneris/lora-aurora">Aurora-IA V1</a>

Dataset utilizado <a href="https://huggingface.co/datasets/chenuneris/aurora-mix-data-baize-format/tree/main">Dataset-V1</a>

Modelo lora v2 <a href="https://huggingface.co/chenuneris/lora-aurorav2">Aurorav2-IA</a>

Dataset aurora v2 <a href="https://huggingface.co/datasets/chenuneris/lora-aurora-v2">Dataset-v2</a>

Modelo ggml v2-ref-doc <a href="https://huggingface.co/chenuneris/aurora-v2-doc-ref">Aurora-v2-doc-ref</a>

Para subir a aplicação com lora (não suporta referencias):

```
Efetue o download do modelo lora citado acima

base_model=decapoda-research/llama-7b-hf
lora_model=project-baize/baize-lora-7B
cd aurora-ia/demo/
python app.py $base_model $lora_model

```

instalar demendencias:
```
pip install -r requirements.txt
```

Para executar um chat simples com os modelos quantizados:
```
# Faça o download do modelo no huggingface.
# Efetue a ingestão dos arquivos para criar uma chromadb.
python ingest.py refs/korean-war-usa-clean.txt
# Execute o chat
python ggml_chat.py -m ./aurorav2-ultrachat-refer-q5_1.bin --cpu 
```

