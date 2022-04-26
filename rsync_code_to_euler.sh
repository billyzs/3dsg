#!/bin/sh
rsync -CPzuhvra --include='**.gitignore' --exclude='/.git' --filter=':- .gitignore' --delete-after ./ euler.ethz.ch:~/delta_3dsg
