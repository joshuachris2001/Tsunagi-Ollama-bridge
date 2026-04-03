../llama.cpp/build/bin/llama-quantize \
             --tensor-type "^v\.blk\.0\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.1\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.2\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.3\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.4\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.5\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.6\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.7\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.8\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.9\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.10\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.11\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.12\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.13\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.14\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.15\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.16\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.17\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.18\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.19\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.20\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.21\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.22\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.23\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.24\.attn_v\.weight=q5_0" \
             --tensor-type "^v\.blk\.25\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.blk\.26\.attn_v\.weight=q8_0" \
             --tensor-type "^v\.=f16" \
             jan-v2-vl-merged.gguf \
             jan-v2-vl-Q4_K_M.gguf \
             Q4_K_M 12
