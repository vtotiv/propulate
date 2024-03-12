#!/bin/bash

# default values
dataset="mnist"
surrogate="default"
seed="42"

# change ip_map to match your server IPs
declare -A ip_map=(
    ["1"]="127.0.0.1"   # bachelor-v100-1
    ["2"]="127.0.0.1"   # bachelor-v100-2
    ["3"]="127.0.0.1"   # bachelor-v100-3
    ["4"]="127.0.0.1"  	# bachelor-v100-4
)
# default ip
ip="${ip_map[1]}"

usage() {
    echo "Usage: $0 --data <mnist|cifar> --surrogate <default|static|dynamic> --seed <42|271|3141> --ip <1|2|3|4>"
    exit 1
}

# parse cli arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data)
            if [[ "$2" == "mnist" || "$2" == "cifar" ]]; then
                dataset="$2"
            else
                usage
            fi
            ;;
        --surrogate)
            if [[ "$2" == "default" || "$2" == "static" || "$2" == "dynamic" ]]; then
                surrogate="$2"
            else
                usage
            fi
            ;;
        --seed)
            if [[ "$2" == "42" || "$2" == "271" || "$2" == "3141" ]]; then
                seed="$2"
            else
                usage
            fi
            ;;
        --ip)
            ip="${ip_map[$2]}"
            if [[ -z "$ip" ]]; then
                echo "Invalid IP shortcut: $2"
                usage
            fi
            ;;
        *) usage ;;
    esac
    shift 2
done

echo "Dataset: $dataset"
echo "Surrogate: $surrogate"
echo "Seed: $seed"
echo "IP: $ip"

scp -i /path/to/private_key -r root@$ip:/home/username/propulate/perun_results "${dataset}_${surrogate}_s${seed}/"
scp -i /path/to/private_key -r root@$ip:/home/username/propulate/${dataset}_log_0.csv "${dataset}_${surrogate}_s${seed}/"
scp -i /path/to/private_key -r root@$ip:/home/username/propulate/${dataset}_log_1.csv "${dataset}_${surrogate}_s${seed}/"

