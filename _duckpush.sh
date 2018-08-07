#!/bin/bash
npm run $1  	# use babel to convert ES6 to ES5 saves to ducksrc area
cp ../service/$1/README.md /media/sf_vmshare/ducksrc/readmes/$1.md
echo "Ready to run jsduck on $1.  Then use duckpull to move jsduck results into doc area."