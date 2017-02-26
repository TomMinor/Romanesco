#!/bin/bash

(./romanesco &) && sleep 1 && xprop -f _GTK_THEME_VARIANT 8u -set _GTK_THEME_VARIANT "dark" -name "Romanesco"
