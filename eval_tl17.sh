TIMELINES=(finan bpoil h1n1 haiti iraq libya mj syria egypt)
CONFS=(ap-abstractive.json ap-extractive.json ap-abstractive-oracle.json extractive-oracle.json)


if test "$#" -eq 0; then
    TRY_CONFS=$CONFS
	for E in "${CONFS[@]}"; do
    	TRY_CONFS+=("configs/${E}")
	done
else
    TRY_CONFS=$@
fi

for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]} 
    do
    python graphsum/tleval.py -t corpora/timeline17/${TL}/timelines/* -- corpora/tl17-${TL}.pkl $CONF
    python graphsum/tleval.py -c tok -t corpora/timeline17/${TL}/timelines/* -- corpora/tl17-${TL}.pkl $CONF
    done
done
