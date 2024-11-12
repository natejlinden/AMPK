#! /bin/zsh

cd /Users/natetest/Documents/phd/research/Project_3_AMPK/src/ampk_models/identifiability

julia ./ampk_Coccimiglio.jl
julia ./MA_single.jl
julia ./MM_single.jl
julia ./MA_nonessential.jl
julia ./MM_nonessential.jl