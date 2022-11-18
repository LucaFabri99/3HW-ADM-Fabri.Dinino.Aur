#!/bin/bash

countries=("Italy" "Spain" "France" "England" "United_States")
for country in ${countries[@]};
do
  country=${country/_/ } # in order to get 'United States' instead of 'United_States'
  echo Country : ${country}

  # We can get the data of the attributes that we need (numPeopleVisited, numPeopleWant, placeAddress) using the function cut.
  # With the command grep we get the data only for the country that we are interested in.
  nb_places=$(cut -f 3,4,8 final_dataset.tsv | grep "${country}" | wc -l)
  echo Number of places that can be found in ${country} : $nb_places


  #We use a counter because we need the average visit of the places, the second cut gets the first column of the columns we already get, which means the 3 column of our dataset
  i=0
  for numPeopleVisited in $(cut -f 3,4,8 final_dataset.tsv | grep "${country}" | cut -f 1); do
    i=$(( $i+$numPeopleVisited ))
  done
  echo Average visit of the places of ${country} : $(( $i/$nb_places ))


   #Here we use a new command awk because we only need the sum of the people that want to visit
    echo People that want to visit the places of ${country} : $(cut -f 3,4,8 final_dataset.tsv | grep "${country}" | awk '{total = total + $2}END{print total}')
  
  echo
done
