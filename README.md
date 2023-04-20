<p align="center">
<img width="500px" alt="Aurora IA" src="">
</p>

# Aurora IA 

Este repositorio utiliza a formatação, scripts e aplicação originais do projeto Baize-Chatbot, porém modificado com objetivo de gerar conteúdo enviesado para uma perspectiva específica. Neste projeto nao utilizamos nenhum dos dados gerados pelo projeto original do Baize.


Para subir a aplicação:

```
tar -xzvf lora-aurora.tar.xz
base_model=decapoda-research/llama-7b-hf
lora_model=project-baize/baize-lora-7B
python app.py $base_model $lora_model
```
