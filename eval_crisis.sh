TIMELINES=(yemen libya syria egypt)
CONFS=(ap-abstractive-trained.json ap-abstractive.json ap-extractive.json ap-abstractive-oracle.json extractive-oracle.json)


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
    python tlgraphsum/tleval.py -t corpora/crisis.data/${TL}/public/timelines/*  -- corpora/crisis-${TL}.pkl $CONF
    done
done


for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]} 
    do
    python tlgraphsum/tleval.py -c tok -t corpora/crisis.data/${TL}/public/timelines/*  -- corpora/crisis-${TL}.pkl $CONF
    done
done
