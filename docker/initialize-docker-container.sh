#!/bin/zsh
# install octo
source ~/.zshrc
cd ~/workspace && pip install -e .

# install rlds_dataset_mod
source ~/.zshrc
cd ~/workspace/rlds_dataset_mod && pip install -e .

# git safe directory
cd ~/workspace && git config --global --add safe.directory /root/workspace

cd ~/.

# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null
