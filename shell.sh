# echo $(date)
# for i in {1..5}
# do
#   python main.py 3 $i
# done
# echo $(date)
# for i in {1..5}
# do
#   python main.py 4 $i
# done
# echo $(date)
# for i in {1..5}
# do
#   python main.py 5 $i
# done
# echo $(date)

for k in {4..4}
do
    for prob in {3..3}
    do
        for t in {1..2}
        do
            echo $(date)
                python main.py $k $prob $t
        done
    done
done