#!/usr/bin/env bash

nvidia-docker run -it -v /home/grad/boer/shared:/root/shared --link facetag-db:facetag-db --name leuko-tf leuko-tf /bin/bash