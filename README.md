# Why?
Sparked from the R&D of brute forcing [Jan-v2-VL](https://ollama.com/fredrezones55/Jan-v2-VL) to work with Ollama, I have found the conflicts where Ollama does not like and created a patcher that simply merges the GGUF models to how Ollama expects. Ollama still has a few kinks, but that is part of the Ollama limitations with how the chat templites are handled; after this work I did not want to hog and Ollama's built-in create HF to Ollama requres a LOT of ram, where as llama.cpp's tools use mmaping [much nicer on memory].

### What's this about a model BLOB?
To ensure the program had proper tensor configurations, I found it was easyer take vital mmproj information from the offical vision tensors than hardcoding it. Mainly to verify the Ollama Vision limits, Attention structure, and RoPE information
Not completely required for Qwen3-VL models as most of it was hardcoded from taking samples from the blob models directly and pulling these values out (the said bruteforcing to get it to work). It seemded between the diffrent model sizes with Qwen3.5, hardcoding would not completly cut it.
You can get the model BLOB, by downloading qwen3.5 model size for the base model; and looking at the resulting modelfile.

# Preparation
In your python enviorment you should have `gguf` installed, I also included `tqdm` for progress bars; we also need `numpy` to handle the arrays.
> pip install gguf tqdm numpy

If we are merging a qwen3.5 model, there are parts of the code that was not hardcoded to get it to work; so we need to download the model your finetune was based off of.

For example Qwen3.5:4B:

4B base qwen3.5 model
> ollama pull qwen3.5:4b
> 
> ollama show --modelfile qwen3.5:4b

should be the second `FROM` to show the full path of the model BLOB on your system.

# Usage
The program has 3-4 important arguments
`model-type` - I have not put an auto discover routine in the program [*I could perhaps pull it from the inputted model*], so the program would have to be told what model architecture this is:
- `qwen3vl` [source blob not required]
- `qwen3vlmoe` [source blob not required]
- `qwen35` [source blob is required]

`llm` - the finetuned text model you want to use.

`mmproj` - the mmproj vision file to merge in with the text model.

`blob` - the source model the qwen3.5 pass needs, as there might be a diffrence between 4B and 27B.

`output` - where the outputted merged gguf model would go, <sub>Note: by merging the model file, it will no longer be supported by llama.cpp</sub>


# Ai Recipt
I used Claude Sonnet 4.6 to assist with the program stucture and the merging of the merge programs.
