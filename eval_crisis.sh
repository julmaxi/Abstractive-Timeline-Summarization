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

FLAGS=""
if [ "$1" == '-f' ]; then
    FLAGS="-f"
fi

for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]}
    do
    python tlgraphsum/tleval.py $FLAGS -t corpora/crisis.data/${TL}/public/timelines/*  -- corpora/crisis-${TL}.pkl $CONF
    done
done


for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]}
    do
    python tlgraphsum/tleval.py $FLAGS -c tok -t corpora/crisis.data/${TL}/public/timelines/*  -- corpora/crisis-${TL}.pkl $CONF
    done
done
