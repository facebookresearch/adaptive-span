#!/bin/bash

set -e

# Usage: ./submit.sh job_name partition ngpus constraint arg1 arg2 ...
name=$1
partition=$2
ngpus=$3
constraint=$4
nodes=$5
args=${@:6}

base_dir=/checkpoint/$USER/jobs/char/$name
mkdir -p $base_dir

codedir=$base_dir/code/
if [ -d "$codedir" ]; then
  read -r -p "The code already exists. Overwrite it? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    rsync -a /private/home/$USER/adaptive-span/ $codedir
  fi
else
  rsync -a /private/home/$USER/adaptive-span/ $codedir
fi

echo "$args" > $base_dir/args.txt
git log|head -6 > $base_dir/git.txt
echo -e "\n\n" >> $base_dir/git.txt
git diff >> $base_dir/git.txt

cd $base_dir/code
host=$(hostname)
./submit.py --name $name --folder $base_dir --partition $partition --ngpu $ngpus --constraint $constraint --nodes $nodes \
  --args "$args --checkpoint $base_dir/model.pt --distributed --plot --plot-host http://${host} --plot-env $name"
