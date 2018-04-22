#!/bin/zsh

# change to home directory
# cd ~

# set zsh as default shell
sudo sed -i 's/required/sufficient/g' /etc/pam.d/chsh
chsh -s $(which zsh)

# install oh-my-zsh
rm -rf ~/.oh-my-zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
sleep 5s
# echo -e "exit\n"

# clone dotfiles repo and setup
rm -rf ~/dotfiles
git clone https://github.com/gtbai/dotfiles.git ~/dotfiles
source ~/dotfiles/setup.sh
source ~/.zshrc
