#!/bin/bash
for f in {'tensorflow','keras','sklearn','numpy','scipy','pandas','matplotlib'}; do echo ; python -c "import  as library_version; print(library_version.__version__)";done
