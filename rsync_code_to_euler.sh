#!/bin/sh
rsync -CPzuhvra --include='**.gitignore' --exclude='/.git' --filter=':- .gitignore' ./ euler.ethz.ch:~/delta_3dsg
