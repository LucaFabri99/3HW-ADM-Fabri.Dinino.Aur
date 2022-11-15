#!/bin/bash

countries=("Italy" "Spain" "France" "England" "United_States")
for country in ${countries[@]};
do
  country=${country/_/ } # in order to obtain 'United States' instead of 'United_States'
  echo Country : ${country}
  
  # We can get the data of the attributes that we need (numPeopleVisited, numPeopleWant, placeAddress) using the function cut.
  # With the command grep we get the data only for the country that we are interested in.
  echo Number of places that can be found in ${country} :
  nb_places=$(cut -f 3,4,8 final_dataset.csv | grep "${country}" | wc -l)
  echo $nb_places

  # col 1 = numPeopleVisited
  i=0
  for numPeopleVisited in $(cut -f 3,4,8 final_dataset.csv | grep "${country}" | cut -f 1); do
    i=$(( $i+$numPeopleVisited ))
  done
  echo Average visit of the places of ${country} :
  i=$(( $i/$nb_places ))
  echo $i
  
  # col 2 = numPeopleWant
  k=0
  for numPeopleWant in $(cut -f 3,4,8 final_dataset.csv | grep "${country}" | cut -f 2); do
    k=$(( $k+$numPeopleWant ))
  done
  echo People that want to visit the places of ${country} :
  echo $k

  echo
done
