#!/bin/bash

#--------PATHS--------
data_path="../kegg-data/"
output_path="../find_pdb/"
coordinates_path="../coordinates/"
pdb_path="../pdb/"
image_path="../image/"
results_path="../results/"

#--------DATA---------
base_name="sequences"
workers=5
epochs=10
samples_by_ec=500
declare -a architectures=("A" "B" "C" "D" "E" "F" "G" "H")
declare -a image_sizes=(256 512)

declare -a tasks=("classification")

if [[ `ls ${data_path}${base_name}[0-9].csv | wc -l` != $workers ]];
then
    python3 split_by_workers.py ${data_path} ${base_name} ${data_path} ${workers}
else
    echo "Data already splitted, skipping."
fi
    
if [[ `ls ${output_path}${base_name}[0-9].csv | wc -l` != $workers || ! -d ${pdb_path} ]];
then
    parallel -u "python3 download_pdb.py ${data_path} '${base_name}{.}' ${output_path}" ::: $( seq 0 $(($workers-1)) )
else
    echo "FindPdb executed, skipping."
fi

if [[ ! -f "${output_path}${base_name}.csv" ]];
then
    python3 merge_outputs.py ${output_path} ${base_name} ${output_path} ${workers}
else
    echo "Outputs merged, skipping."
fi

#if [[ ! -d ${coordinates_path} ]];
#then
#    python3 get_coordinates.py ${output_path}${base_name}.csv ${pdb_path} ${coordinates_path} 3500
#else
#    echo "Coordinates already created, skipping."
#fi

for size in ${image_sizes[@]}; do
    if [[ ! -d "${image_path}${size}/" ]];
    then
        mkdir -p ${image_path}
        python3 get_images.py ${output_path}${base_name}.csv ${pdb_path} "${image_path}${size}/" ${size} ${samples_by_ec}
    else
        echo "Images ${size} already generated, skipping."
    fi
done

mkdir -p ${results_path}

for size in ${image_sizes[@]}; do
    for architecture in ${architectures[@]}; do
        for task in ${tasks[@]}; do
            if [[ ! -f "${results_path}results_CNN_${architecture}/results_size-${size}_scale-${scale}_task-${task}.json" ]];
            then       
                python3 cnn_3d.py ${image_path}${size}/ ${size} ${architecture} "${results_path}results_CNN_${architecture}/" 0 ${epochs} ${task}
                python3 cnn_3d.py ${image_path}${size}/ ${size} ${architecture} "${results_path}results_CNN_${architecture}/" 1 ${epochs} ${task}
            fi
        done
    done
done
