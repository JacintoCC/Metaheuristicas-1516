#!/bin/bash

#KNN
python Practica2.py W -a K -write True
python Practica2.py L -a K -write True
python Practica2.py A -a K -write True

#SFS
python Practica2.py W -a S -write True
python Practica2.py L -a S -write True
python Practica2.py A -a S -write True

#BMB
python Practica2.py W -a B -write True
python Practica2.py L -a B -write True
python Practica2.py A -a B -write True

#GRASP
python Practica2.py W -a G -write True
python Practica2.py L -a G -write True
python Practica2.py A -a G -write True

#ILS
python Practica2.py W -a I -write True
python Practica2.py L -a I -write True
python Practica2.py A -a I -write True
