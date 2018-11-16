#!/bin/bash
for f in {'tensorflow','keras','sklearn','numpy','scipy','pandas','matplotlib'};do 
  echo $f
  python -c "import $f as library_version; print(library_version.__version__)"
 done
