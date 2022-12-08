#!/bin/bash
set -xe

function main {
    # set common info
    source common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    pip uninstall deepctr -y
    pip install deepctr==0.9.2
    cd examples

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            #cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            #mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
	    python run_deepfefm.py --evaluate \
		--epochs 20 --num_iter ${num_iter} --num_warmup 3 \
		--precision ${precision} --batch_size $batch_size \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
timeout 10s wget -q -O common.sh https://raw.githubusercontent.com/mengfei25/oob-common/gpuoob/common.sh || cp /tmp/common.sh .

# Start
main "$@"
