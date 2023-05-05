<p align="center">
<img width="500px" alt="Aurora IA" src="https://i.redd.it/zkndu5kilmc31.png">
</p>

# Aurora IA 

Este repositorio utiliza a formatação, scripts e aplicação originais do projeto <a href="https://github.com/project-baize/baize-chatbot">Baize-Chatbot</a>, porém modificado com objetivo de gerar conteúdo enviesado para uma perspectiva específica. Neste projeto nao utilizamos nenhum dos dados gerados pelo projeto original do Baize.


Modelo lora  <a href="https://huggingface.co/chenuneris/lora-aurora">Aurora-IA</a>

Dataset utilizado <a href="https://huggingface.co/datasets/chenuneris/aurora-mix-data-baize-format/tree/main">Dataset</a>

Modelo lora v2 <a href="https://huggingface.co/chenuneris/lora-aurorav2">Aurorav2-IA</a>

Dataset aurora v2 <a href="https://huggingface.co/datasets/chenuneris/lora-aurora-v2">Dataset-v2</a>

Para subir a aplicação:

```
tar -xzvf lora-aurora.tar.xz
base_model=decapoda-research/llama-7b-hf
lora_model=project-baize/baize-lora-7B
python app.py $base_model $lora_model
```
