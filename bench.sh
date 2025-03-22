#!/bin/bash

T_values=(16 32 64)
write_ratio_values=(0.05 0.1 0.2 0.5 0.8 0.9)

BASE_CMD="./apps/concurrent_bench --data_type float --dist_fn l2 --data_path data/sift/sift_base.fbin --query_file data/sift/sift_query.fbin --gt_file data/sift/sift_query_base_gt100 -R 32 --Lbuild 40 --Ls 40 --alpha 1.2 -L 10 20 30 40 50 100"

for T in "${T_values[@]}"; do
    for write_ratio in "${write_ratio_values[@]}"; do
        CMD="$BASE_CMD -T $T --write_ratio $write_ratio"
        
        echo "Running: T=$T, write_ratio=$write_ratio"
        $CMD
        
        if [ $? -eq 0 ]; then
            echo "Completed: T=$T, write_ratio=$write_ratio"
        else
            echo "Error: T=$T, write_ratio=$write_ratio failed"
        fi
        
        sleep 2
    done
done

echo "All benchmarks completed!"