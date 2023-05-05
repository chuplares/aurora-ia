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
Efetue o download do modelo lora citado acima

base_model=decapoda-research/llama-7b-hf
lora_model=project-baize/baize-lora-7B
cd aurora-ia/demo/
python app.py $base_model $lora_model

```

Para executar um chat simples com os modelos quantizados:
```
Faça o download dos modelos quantizados nos links citados acima e grave o caminho absoluto do arquivo
Clone o repositorio do llama.cpp
git clone llama.cpp
Acesse o diretorio
cd llama.cpp
Compile os arquivos para gerar os binarios
make
faça o download dos modelos quantizados nos links citados acima
Acesse a pasta onde os binarios compilados estão localizados
cd build/bin
Execute o chat utilizando o prompt disponibilizado no repositorio
./main -m /caminho/modelo/aurora-v2-q5-1-ggml.bin --top_p 0.95 --frequency_penalty 1.1 -s 42 -c 2048 -n 2048 -i -r "[|Human|]" --in-prefix " " -f /caminho/aurora-ai/prompts/aurorav2.txt

```
