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

FLAGS=""
if [[ "$1" == '-f' ]]; then
    FLAGS="-f"
    TRY_CONFS=${@:2}
fi


for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]}
    do
    python tlgraphsum/tleval.py $FLAGS -t corpora/timeline17/${TL}/timelines/* -- corpora/tl17-${TL}.pkl $CONF
    done
done


for CONF in ${TRY_CONFS[@]}
do
    for TL in ${TIMELINES[@]}
    do
    python tlgraphsum/tleval.py $FLAGS -c tok -t corpora/timeline17/${TL}/timelines/* -- corpora/tl17-${TL}.pkl $CONF
    done
done
