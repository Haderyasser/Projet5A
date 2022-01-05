Le but de cet algorithme est de masquer le sol sur différentes types d'image de végétation en utilisant des indices de végétation. 

Pour lancer le masking il faut mettre les images dans un dossier et lancer dans un terminal la commande suivante: 

**python3 masking_vegetaion.py images/ output/**

Un dossier sera crée avec le mask binaire et l'image masquée

## Pour aller plus loin:
La méthode peut être résumer en quelques étapes: 

1. Segmentation de l'image à l'aide de Kmeans (avec une initialisation Kmeans++ et une distance L2)

2. Binarisation des labels à l'aide d'un Kmeans appliqué sur les centroïdes estimés à l'étape 1

3. Reduction du bruit en appliquant des opérateurs morphologiques (une érosion + une dilatation)
Une approche par patch peut être utilisée.


### _Masking_kmeans_patch.py_ contient la fonction principale: mask_kmeans dont les paramètres sont: 
- img: image RGB
- index_vg: indice de végétation
- Ncluster: nombre de clusters pour Kmeans
- morph_kernel_size: la taille du kernel pour l'érosion et la dilatation
- dilat : la taille du kernel qui permet de dilater d'avantage le masque
- r x c: le nombre de patches

**NB**: Le choix de ces paramètres peut se faire à travers la fonction: _kmeans_parameters_ qui prend comme paramètres l'image et le type de végétation

### _vegetation_index.py_ calcule les différentes types d'indices de végétation
- ExG: 
- NDI: 
- a_lab:
- mean: moyenne des 3 précédents indices





