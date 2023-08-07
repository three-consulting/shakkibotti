#!/bin/bash

wget -O games.pgn.zst $1
unzst -o games.pgn games.pgn.zst
rm games.pgn.zst

source env/bin/activate
python preprocess.py preprocess --files games.pgn --outDir $2
python preprocess.py tokenize --outDir $2
rm $2/moves_*