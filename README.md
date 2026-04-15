# 繋
### Project GGUF patcher for Ollama Multimodal Monoliths
This project will take compatible models' pure llama.cpp text and mmproj models and convert it to a Ollama GGUF Monolith model.

This project is modular, Instead of it's predicasor of spegetti code 🍝; it would be now easier to add model types and multimodal functions. 🧩

### What is a Monolith model?
Ollama's new engine handles GGUF models diffrently; these models are taken from hugging face models and converted to a **Merged** gguf format, this reduces memory overhead by storing the multimodal tensors in one place.
This is why pure gguf models crash Ollama when acumpanied with a mmproj; it's formatted wrong or uses the wrong tensor formatting.

### Orgins
* Qwen3.x GGUF vision patcher; This was simple project made to quickly patch the slightly buggy Qwen3-VL and Qwen3.5 model family in Ollama. Made for people who would prefer Ollama over llama.cpp [*I do wish that Ollama fixes the template handling, but what can we do~*]. [*and little RAM use due to mmap-ing*]

Sparked from the need to find other ways to create Ollama GGUF multimodal models where limited memory is involved; (Will work on lessening the reliance of the source BLOBS over time); default `Ollama create` typicaly requires the enitre model loaded into RAM and can cause OOM kills; my system could not handle this, so I had to come up with a quick alternitve. GGUF is used here to reroute tensors to expectations [work on quant matching later], these tools use mmaping which reduces the RAM draw considerably.

R&D of brute forcing [Jan-v2-VL](https://ollama.com/fredrezones55/Jan-v2-VL) to work with Ollama, I found the conflicts where Ollama does not like and created a patcher that merges the GGUF models to how Ollama expects. Ollama still has a few kinks, but that is part of the Ollama limitations with how the chat templates are handled.

### What's this about a model BLOB?
To ensure the program had proper tensor configurations, I found it was easier to take vital mmproj and other tensor information from the official vision tensors than hardcoding it — mainly to verify the Ollama vision limits, attention structure, and RoPE information, etc.

You can get the needed model BLOB by downloading the model size for that particular base model; 

**This is important:** _For most cases the model of the finetuned model you have needs to be the same as the Ollama GGUF BLOB._

## Preparation
In your Python environment you should have `gguf` installed. I also included `tqdm` for progress bars; we also need `numpy` to handle the arrays.
```
pip install gguf tqdm numpy
```

If we are merging a qwen3.5 model, there are parts of the code that were not hardcoded to get it to work, so we need to download the model your finetune was based off of.

For example Qwen3.5 4B:

```bash
ollama pull qwen3.5:4b
ollama show --modelfile qwen3.5:4b
```

The second `FROM` line will show the full path of the model BLOB on your system.

## Usage
The program has 3–4 important arguments.

`--model-type` — The program needs to be told what model architecture this is (auto-discovery is not yet implemented):
- `qwen3vl` — source blob not required
- `qwen3vlmoe` — source blob not required
- `qwen35` — source blob is required
- `gemma4` — source blob is required 

`--llm` — the finetuned text model you want to use.

`--mmproj` — the mmproj vision file to merge in with the text model.

`--blob` — the source model blob the qwen3.5 path needs, as there may be differences between 4B and 27B.

`--output` — where the output merged GGUF model will go. *(Note: by merging the model file, it will no longer be supported by llama.cpp)*

### Examples

**Qwen3-VL finetune (no blob needed):**
```bash
python OllamaGGUFMerge.py \
    --model-type qwen3vl \
    --llm    my-finetune.Q5_K_M.gguf \
    --mmproj mmproj.gguf \
    --output merged_model.gguf
```

**Qwen3-VL-MOE finetune (no blob needed):**
```bash
python OllamaGGUFMerge.py \
    --model-type qwen3vlmoe \
    --llm    my-finetune.Q4_K_M.gguf \
    --mmproj mmproj.gguf \
    --output merged_model.gguf
```

**Qwen3.5 finetune (blob required):**
```bash
python OllamaGGUFMerge.py \
    --model-type qwen35 \
    --blob   /var/lib/ollama/blobs/sha256-81fb60... \
    --llm    my-finetune.Q6_K.gguf \
    --mmproj mmproj.gguf \
    --output merged_qwen35.gguf
```

If `--output` is omitted, the merged model will be saved as `merged_qwen.gguf` in the current working directory.

# AI Receipt
I used Claude Sonnet 4.6 (via Perplexity) to assist with the program structure.
