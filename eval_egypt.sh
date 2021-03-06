TIMELINES=(egypt)
CONFS=(ap-abstractive-trained.json ap-abstractive.json ap-extractive.json ap-abstractive-oracle.json extractive-oracle.json)


if test "$#" -eq 0; then
    TRY_CONFS=$CONFS
else
    TRY_CONFS=$@
fi

for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]} 
    do
    python graphsum/tleval.py -t corpora/${TL}/timelines/* -- corpora/tl17-${TL}.pkl configs/$CONF
    python graphsum/tleval.py -c tok -t corpora/${TL}/timelines/* -- corpora/tl17-${TL}.pkl configs/$CONF
    done
done
